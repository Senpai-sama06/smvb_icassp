import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer

class Variable_Control_CDR:
    """CDR capable of using either Epsilon or Kappa as the control variable"""
    def __init__(self, num_mics=4, control_type='epsilon', kappa_shift=25.0, zeta_min=1e-5, zeta_max=0.5):
        self.M = num_mics
        self.control_type = control_type
        
        # Sigmoid parameters
        self.sigmoid_kappa = kappa_shift 
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ulb_engine = UniversalLinearBeamformer(num_mics=self.M)
        
        # Thresholds
        self.epsilon_0 = 1.0 / self.M
        # A typical normalized threshold for Condition Number (Log10 scale)
        self.kappa_0 = 3.0 

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        evals, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        trace_R = np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12
        
        if self.control_type == 'epsilon':
            # Proposed: Spatially Informed (Geometric Leakage)
            d_exp = target_rtf[:, np.newaxis, :]
            num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
            d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
            u1_norm = np.linalg.norm(u1, axis=-1)
            rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
            epsilon = 1.0 - (rho ** 2)
            
            # Map Epsilon
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.sigmoid_kappa * (epsilon - self.epsilon_0)))
            
        elif self.control_type == 'kappa':
            # Literature Baseline: Conditioning-Aware (Eigenvalue Spread)
            lambda_max = np.abs(evals[..., -1])
            lambda_min = np.abs(evals[..., 0]) + 1e-12
            cond_num = lambda_max / lambda_min
            log_kappa = np.log10(cond_num + 1e-12)
            
            # Map Log Kappa (using a scaled transition)
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-2.0 * (log_kappa - self.kappa_0)))

        zeta_tensor = z_factor * trace_R
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        return self.ulb_engine.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)

    def apply_weights(self, stft_mix, weights):
        return self.ulb_engine.apply_weights(stft_mix, weights)

def compute_weight_jitter(weights):
    delta_w = weights[:, 1:, :] - weights[:, :-1, :]
    l2_sq = np.sum(np.abs(delta_w)**2, axis=-1) 
    return np.mean(l2_sq)

def calculate_sisdr(target, est, mix):
    min_len = min(len(target), len(est), len(mix))
    t, e, m = target[:min_len], est[:min_len], mix[:min_len]
    def si_sdr(ref, sig):
        ref = ref - np.mean(ref); sig = sig - np.mean(sig)
        alpha = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-12)
        target_scaled = alpha * ref
        noise_sig = sig - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise_sig, noise_sig) + 1e-12))
    return si_sdr(t, e) - si_sdr(t, m)

def main():
    print("--- RUNNING MISSING EXP 6: EPSILON VS KAPPA BASELINE ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    M = 4 
    
    # Simulate a dynamic scenario (Target 90, Encroaching Interferer 85)
    print(">>> Simulating Dynamic Acoustic Scene <<<")
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    mix, target, _, _ = simulator.simulate(
        n=1, reverb=True, target_rt60=0.2, interferer_angles=[85], save_outputs=False
    )
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
    
    modes = ['epsilon', 'kappa']
    labels = ['Proposed Spatially Informed\n(Control = $\epsilon$)', 'Conditioning-Aware Baseline\n(Control = $\kappa(\mathbf{R}_y)$)']
    
    sisdr_results = []
    jitter_results = []
    
    for mode in modes:
        cdr = Variable_Control_CDR(num_mics=M, control_type=mode)
        weights = cdr.process(R_matrix, oracle_rtf)
        est = librosa.istft(cdr.apply_weights(mix_stft, weights), hop_length=hop_length)
        
        sisdr = calculate_sisdr(target[0], est, mix[0])
        jitter = compute_weight_jitter(weights)
        
        sisdr_results.append(sisdr)
        jitter_results.append(jitter)
        print(f"[{mode.upper()}]: SI-SDRi = {sisdr:.2f} dB, Jitter = {jitter:.6f}")

    # --- Plotting the Comparison ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['crimson', 'darkgray']
    
    # Panel 1: Selectivity (SI-SDR)
    axs[0].bar(labels, sisdr_results, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axs[0].set_title('(a) Signal Separation (SI-SDRi)', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('SI-SDR Improvement (dB)', fontsize=12)
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Panel 2: Stability (Jitter)
    axs[1].bar(labels, jitter_results, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axs[1].set_yscale('log')
    axs[1].set_title('(b) Temporal Filter Jitter ($\|\Delta \mathbf{w}\|_2^2$)', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Mean Frame-to-Frame Variance (Log Scale)', fontsize=12)
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.suptitle('Proposed Control Variable ($\epsilon$) vs. Literature Baseline ($\kappa$)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp6_epsilon_vs_kappa_baseline.png', dpi=300)
    print("\n✅ Saved 'results/exp6_epsilon_vs_kappa_baseline.png'")

if __name__ == "__main__":
    main()
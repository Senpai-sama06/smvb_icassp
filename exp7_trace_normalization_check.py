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

class Normalized_Trace_CDR:
    """CDR testing Full Trace vs Mean Trace (Trace / M)"""
    def __init__(self, num_mics=4, trace_mode='full', epsilon_0=None):
        self.M = num_mics
        self.trace_mode = trace_mode
        self.epsilon_0 = epsilon_0 if epsilon_0 else 1.0 / self.M
        self.kappa = 25.0
        self.zeta_min = 1e-5
        self.zeta_max = 0.5
        self.ulb_engine = UniversalLinearBeamformer(num_mics=self.M)

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        _, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        d_exp = target_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        
        # --- THE CORE DIFFERENCE ---
        if self.trace_mode == 'full':
            trace_power = np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12
        elif self.trace_mode == 'mean':
            trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / self.M
            
        z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.kappa * (epsilon - self.epsilon_0)))
        zeta_tensor = z_factor * trace_power
        
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        return self.ulb_engine.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)

    def apply_weights(self, stft_mix, weights):
        return self.ulb_engine.apply_weights(stft_mix, weights)

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
    print("--- RUNNING MISSING EXP 7: TRACE NORMALIZATION CHECK ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    array_sizes = [2, 4, 8]
    test_angle = 45 
    
    modes = ['full', 'mean']
    results = {mode: [] for mode in modes}
    
    for M in array_sizes:
        print(f"\n>>> Simulating {M}-Microphone Array <<<")
        simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=20, fs=fs)
        evaluator = Evaluator(ref_mic=0)
        
        i_angles = [test_angle]
        if M == 4: i_angles = [test_angle, 150, 240]
        elif M == 8: i_angles = [test_angle, 120, 150, 180, 210, 240, 300]
            
        mix, target, _, _ = simulator.simulate(n=M-1, reverb=True, target_rt60=0.2, interferer_angles=i_angles, save_outputs=False)
        mix = mix.T; target = target.T
        
        mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
        
        for mode in modes:
            cdr = Normalized_Trace_CDR(num_mics=M, trace_mode=mode)
            weights = cdr.process(R_matrix, oracle_rtf)
            est = librosa.istft(cdr.apply_weights(mix_stft, weights), hop_length=hop_length)
            results[mode].append(calculate_sisdr(target[0], est, mix[0]))

    # --- Plotting ---
    x = np.arange(len(array_sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(x - width/2, results['full'], width, label=r'Full Trace: $\zeta \cdot \mathrm{Tr}(\mathbf{R})$ (Current)', color='crimson', edgecolor='black')
    ax.bar(x + width/2, results['mean'], width, label=r'Mean Trace: $\zeta \cdot \frac{\mathrm{Tr}(\mathbf{R})}{M}$', color='steelblue', edgecolor='black')

    ax.set_ylabel('SI-SDR Improvement (dB)', fontsize=12)
    ax.set_title('Impact of Trace Normalization on the 1/M Rule', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}' for m in array_sizes], fontsize=12)
    ax.legend(loc='lower center', fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp7_trace_normalization_check.png', dpi=300)
    print("\n✅ Saved 'results/exp7_trace_normalization_check.png'")

if __name__ == "__main__":
    main()
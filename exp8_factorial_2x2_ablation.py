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

class Factorial_Ablation_CDR:
    def __init__(self, num_mics=4, control='epsilon', policy='continuous', zeta_min=1e-5, zeta_max=0.5):
        self.M = num_mics
        self.control = control      # 'epsilon', 'kappa', 'static'
        self.policy = policy        # 'continuous', 'hard', 'nominal', 'sledgehammer'
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)
        
        # Shared Transition Parameters
        self.slope = 25.0
        self.epsilon_0 = 1.0 / self.M

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        evals, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        # MEAN TRACE NORMALIZATION (Step 1 Locked)
        trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / self.M
        
        z_factor = np.full((F, T), self.zeta_min, dtype=np.float32)

        if self.control == 'static':
            if self.policy == 'nominal':
                z_factor.fill(self.zeta_min)
            elif self.policy == 'sledgehammer':
                z_factor.fill(self.zeta_max)
        else:
            if self.control == 'epsilon':
                d_exp = target_rtf[:, np.newaxis, :]
                num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
                d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
                u1_norm = np.linalg.norm(u1, axis=-1)
                rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
                metric = 1.0 - (rho ** 2)
                threshold = self.epsilon_0
                
            elif self.control == 'kappa':
                lambda_max = np.abs(evals[..., -1])
                lambda_min = np.abs(evals[..., 0]) + 1e-12
                log_kappa = np.log10(lambda_max / lambda_min + 1e-12)
                
                # FAIR CALIBRATION: Min-Max normalize log_kappa to [0, 1] 
                # and set threshold at the median to match Epsilon's dynamic range.
                k_min, k_max = np.percentile(log_kappa, 5), np.percentile(log_kappa, 95)
                metric = np.clip((log_kappa - k_min) / (k_max - k_min + 1e-12), 0.0, 1.0)
                threshold = np.median(metric)

            # Apply Adaptation Policy
            if self.policy == 'hard':
                z_factor = np.where(metric > threshold, self.zeta_max, self.zeta_min)
            elif self.policy == 'continuous':
                z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.slope * (metric - threshold)))

        zeta_tensor = z_factor * trace_power
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        weights = self.ulb.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        
        return weights, z_factor

    def apply_weights(self, stft_mix, weights):
        return self.ulb.apply_weights(stft_mix, weights)

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
    print("--- RUNNING EXP 8: FACTORIAL 2x2 ABLATION ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4 
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    # Severe Encroachment Scenario (forces controllers to work hard)
    print(">>> Simulating Encroachment (Target 90, Interferer 75) <<<")
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False)
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    configs = [
        ('static', 'nominal', 'Nominal MVDR'),
        ('static', 'sledgehammer', 'Fixed Heavy'),
        ('kappa', 'hard', '$\kappa$ + Hard'),
        ('kappa', 'continuous', '$\kappa$ + Continuous'),
        ('epsilon', 'hard', '$\epsilon$ + Hard'),
        ('epsilon', 'continuous', '$\epsilon$ + Continuous (CDR)')
    ]
    
    metrics = {'label': [], 'sisdr': [], 'jitter': [], 'p_sat': [], 'std_z': []}
    
    for ctrl, pol, lbl in configs:
        cdr = Factorial_Ablation_CDR(num_mics=M, control=ctrl, policy=pol)
        weights, z_factor = cdr.process(R_matrix, oracle_rtf)
        est = librosa.istft(cdr.apply_weights(mix_stft, weights), hop_length=hop_length)
        
        sisdr = calculate_sisdr(target[0], est, mix[0])
        jitter = np.mean(np.sum(np.abs(weights[:, 1:, :] - weights[:, :-1, :])**2, axis=-1))
        p_sat = np.mean(z_factor > 0.9 * 0.5) * 100 # Percentage
        std_z = np.std(z_factor)
        
        metrics['label'].append(lbl)
        metrics['sisdr'].append(sisdr)
        metrics['jitter'].append(jitter)
        metrics['p_sat'].append(p_sat)
        metrics['std_z'].append(std_z)
        
        print(f"[{lbl}]: SI-SDR={sisdr:.2f}, Jitter={jitter:.5f}, P_sat={p_sat:.1f}%, std(z)={std_z:.4f}")

    # --- Plotting 4-Panel Results ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['gray', 'black', 'lightblue', 'steelblue', 'lightcoral', 'crimson']
    
    axs[0, 0].bar(metrics['label'], metrics['sisdr'], color=colors, edgecolor='black')
    axs[0, 0].set_title('(a) Speech Quality (SI-SDRi)', fontweight='bold')
    axs[0, 0].set_ylabel('dB')
    
    axs[0, 1].bar(metrics['label'], metrics['p_sat'], color=colors, edgecolor='black')
    axs[0, 1].set_title(r'(b) Saturation Fraction ($P_{sat} > 90\%$)', fontweight='bold')
    axs[0, 1].set_ylabel('% of frames')
    
    axs[1, 0].bar(metrics['label'], metrics['std_z'], color=colors, edgecolor='black')
    axs[1, 0].set_title(r'(c) Regularization Variance ($\operatorname{std}(\zeta)$)', fontweight='bold')
    axs[1, 0].set_ylabel('Standard Deviation')
    
    axs[1, 1].bar(metrics['label'], metrics['jitter'], color=colors, edgecolor='black')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title(r'(d) Filter Jitter ($\|\Delta \mathbf{w}\|_2^2$)', fontweight='bold')
    axs[1, 1].set_ylabel('Log Scale')
    
    for ax in axs.flat:
        ax.set_xticks(range(len(metrics['label'])))
        ax.set_xticklabels(metrics['label'], rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp8_factorial_2x2_ablation.png', dpi=300)
    print("\n✅ Saved 'results/exp8_factorial_2x2_ablation.png'")

if __name__ == "__main__":
    main()
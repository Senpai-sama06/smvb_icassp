import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pathlib import Path

# --- IEEE Formatting Tweaks ---
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer

class Mapping_Ablation_CDR:
    def __init__(self, num_mics=4, mapping='sigmoid'):
        self.M = num_mics
        self.mapping = mapping
        self.epsilon_0 = 1.0 / self.M
        self.zeta_min = 1e-5
        self.zeta_max = 0.5
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)
        self.slope = 25.0

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        evals, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        d_exp = target_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        
        trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / self.M
        
        if self.mapping == 'sigmoid':
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.slope * (epsilon - self.epsilon_0)))
        elif self.mapping == 'linear':
            # Linear ramp with a width of 0.2 centered at epsilon_0
            width = 0.1 
            linear_interp = (epsilon - (self.epsilon_0 - width)) / (2 * width)
            linear_interp = np.clip(linear_interp, 0.0, 1.0)
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) * linear_interp

        zeta_tensor = z_factor * trace_power
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        weights = self.ulb.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        return weights

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
    print("--- GENERATING FIG 5: LINEAR VS SIGMOID MAPPING ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False)
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    modes = ['linear', 'sigmoid']
    labels = ['Linear Mapping', 'CDR-MVDR\n(Sigmoid)']
    colors = ['#4682b4', '#DC143C'] 
    
    res_sisdr = []
    res_jitter = []
    
    for mode in modes:
        cdr = Mapping_Ablation_CDR(num_mics=M, mapping=mode)
        weights = cdr.process(R_matrix, oracle_rtf)
        est = librosa.istft(cdr.apply_weights(mix_stft, weights), hop_length=hop_length)
        
        res_sisdr.append(calculate_sisdr(target[0], est, mix[0]))
        res_jitter.append(np.mean(np.sum(np.abs(weights[:, 1:, :] - weights[:, :-1, :])**2, axis=-1)))

    # --- Plotting 3-Panel Layout ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Panel (a): The Transfer Functions
    eps_vals = np.linspace(0, 1, 500)
    eps_0 = 1.0 / M
    z_min, z_max, slope = 1e-5, 0.5, 25.0
    
    z_sig = z_min + (z_max - z_min) / (1.0 + np.exp(-slope * (eps_vals - eps_0)))
    
    width = 0.1
    linear_interp = (eps_vals - (eps_0 - width)) / (2 * width)
    linear_interp = np.clip(linear_interp, 0.0, 1.0)
    z_lin = z_min + (z_max - z_min) * linear_interp

    # SHORTENED LABELS TO KEEP THE BOX COMPACT
    axs[0].plot(eps_vals, z_lin, color=colors[0], linewidth=3, label='Linear Mapping', zorder=3)
    axs[0].plot(eps_vals, z_sig, color=colors[1], linewidth=3, label='CDR-MVDR', zorder=3)
    axs[0].axvline(eps_0, color='black', linestyle=':', linewidth=2, label=r'Boundary $\epsilon_0$', zorder=2)
    axs[0].set_title(r'(a) $\epsilon$-to-$\zeta$ mappings', fontweight='bold')
    axs[0].set_xlabel(r'Spatial Subspace Leakage $\epsilon$')
    axs[0].set_ylabel(r'Loading Multiplier $\zeta$')
    axs[0].grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # MOVED LEGEND TO THE BOTTOM RIGHT
    axs[0].legend(loc='lower right', framealpha=0.95, edgecolor='black')
    
    # Panel (b): SI-SDRi
    axs[1].bar(labels, res_sisdr, color=colors, edgecolor='black', width=0.5, zorder=3)
    axs[1].set_title('(b) Speech Quality (SI-SDRi)', fontweight='bold')
    axs[1].set_ylabel('dB')
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # Panel (c): Filter Jitter
    axs[2].bar(labels, res_jitter, color=colors, edgecolor='black', width=0.5, zorder=3)
    axs[2].set_yscale('log')
    axs[2].set_title(r'(c) Filter Jitter ($\|\Delta \mathbf{w}\|_2^2$)', fontweight='bold')
    axs[2].set_ylabel('Log Scale')
    axs[2].grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/fig5_linear_vs_sigmoid.pdf', dpi=300, bbox_inches='tight')
    print("\n✅ Saved 'results/fig5_linear_vs_sigmoid.pdf'")

if __name__ == "__main__":
    main()
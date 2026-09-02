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
    'figure.titlesize': 13
})

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer

class Distribution_CDR:
    def __init__(self, num_mics=4, control='epsilon'):
        self.M = num_mics
        self.control = control
        self.zeta_min = 1e-5
        self.zeta_max = 0.5
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)
        self.slope = 25.0
        self.epsilon_0 = 1.0 / self.M

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        evals, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
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
            k_min, k_max = np.percentile(log_kappa, 5), np.percentile(log_kappa, 95)
            metric = np.clip((log_kappa - k_min) / (k_max - k_min + 1e-12), 0.0, 1.0)
            threshold = np.median(metric)

        z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.slope * (metric - threshold)))
        return z_factor

def main():
    print("--- GENERATING FIG 3: LOADING DISTRIBUTION (CDF) ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4 
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    # Severe encroachment (15 degrees separation) to force decisive adaptation
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False)
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    cdr_eps = Distribution_CDR(num_mics=M, control='epsilon')
    z_eps = cdr_eps.process(R_matrix, oracle_rtf).flatten()
    
    cdr_kap = Distribution_CDR(num_mics=M, control='kappa')
    z_kap = cdr_kap.process(R_matrix, oracle_rtf).flatten()
    
    z_eps_sorted = np.sort(z_eps)
    p_eps = 1.0 * np.arange(len(z_eps)) / (len(z_eps) - 1)
    
    z_kap_sorted = np.sort(z_kap)
    p_kap = 1.0 * np.arange(len(z_kap)) / (len(z_kap) - 1)
    
    sat_thresh = 0.9 * 0.5 
    sat_eps = np.mean(z_eps > sat_thresh) * 100
    sat_kap = np.mean(z_kap > sat_thresh) * 100

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot ECDFs with updated labels linking ablation terms to final system names
    ax.plot(z_kap_sorted, p_kap, color='#888888', linewidth=3.5, label=r'$\kappa$-Continuous Controller', zorder=3)
    ax.plot(z_eps_sorted, p_eps, color='#DC143C', linewidth=3.5, label=r'$\epsilon$-Continuous (CDR-MVDR)', zorder=3)
    
    # Highlight Saturation Zone
    ax.axvspan(sat_thresh, 0.505, color='red', alpha=0.08, label=r'Saturation Zone ($> 0.9 \zeta_{max}$)', zorder=1)
    ax.axvline(sat_thresh, color='red', linestyle=':', linewidth=2, zorder=2)
    
    # Annotations right-aligned inside the safe empty space to prevent clipping
    ax.text(0.495, 0.65, f"$\kappa$ Saturated: {sat_kap:.1f}%", color='black', fontweight='bold', fontsize=10, 
            ha='right', va='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#888888', boxstyle='round,pad=0.3'), zorder=4)
    ax.text(0.495, 0.55, f"$\epsilon$ (CDR) Saturated: {sat_eps:.1f}%", color='#DC143C', fontweight='bold', fontsize=10, 
            ha='right', va='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#DC143C', boxstyle='round,pad=0.3'), zorder=4)
    
    ax.set_title('Cumulative Distribution of Adaptive Loading ($\zeta$)', fontweight='bold')
    ax.set_xlabel(r'Regularization Multiplier $\zeta$', fontweight='bold')
    ax.set_ylabel(r'Cumulative Probability $P(\zeta \leq z)$', fontweight='bold')
    
    ax.set_xlim([0, 0.505])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    # Changed to explicitly save as PDF
    plt.savefig('results/fig3_loading_distribution.pdf', dpi=300, bbox_inches='tight')
    print("\n✅ Saved 'results/fig3_loading_distribution.pdf'")

if __name__ == "__main__":
    main()
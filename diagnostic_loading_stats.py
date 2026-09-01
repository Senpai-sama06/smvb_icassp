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
    'legend.fontsize': 10
})

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator

def main():
    print("--- RUNNING DIAGNOSTIC: LOADING STATISTICS ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4 
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    # 15 degrees separation (dangerous scenario)
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False)
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    # FIX: Ensure oracle_rtf is (Frequencies, Mics) for broadcasting
    if oracle_rtf.shape[0] == M and oracle_rtf.shape[1] != M:
        oracle_rtf = oracle_rtf.T
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    _, evecs = np.linalg.eigh(R_matrix)
    u1 = evecs[..., -1]
    
    # Epsilon Math
    d_exp = oracle_rtf[:, np.newaxis, :]
    num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
    d_norm = np.linalg.norm(oracle_rtf, axis=-1)[:, np.newaxis]
    u1_norm = np.linalg.norm(u1, axis=-1)
    rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
    metric_eps = 1.0 - (rho ** 2)
    
    # Kappa Math
    evals, _ = np.linalg.eigh(R_matrix)
    lambda_max = np.abs(evals[..., -1])
    lambda_min = np.abs(evals[..., 0]) + 1e-12
    log_kappa = np.log10(lambda_max / lambda_min + 1e-12)
    k_min, k_max = np.percentile(log_kappa, 5), np.percentile(log_kappa, 95)
    metric_kap = np.clip((log_kappa - k_min) / (k_max - k_min + 1e-12), 0.0, 1.0)
    
    # Map to Continuous Policy
    z_min, z_max, slope = 1e-5, 0.5, 25.0
    z_eps = z_min + (z_max - z_min) / (1.0 + np.exp(-slope * (metric_eps - (1.0/M))))
    z_kap = z_min + (z_max - z_min) / (1.0 + np.exp(-slope * (metric_kap - np.median(metric_kap))))
    
    z_eps = z_eps.flatten()
    z_kap = z_kap.flatten()
    
    # --- CALCULATE EXACT STATS FOR ADVISOR ---
    print("\n--- RESULTS ---")
    print(f"Epsilon Controller: E[z] = {np.mean(z_eps):.4f} | Median = {np.median(z_eps):.4f} | P(z > 0.9*z_max) = {np.mean(z_eps > 0.45)*100:.1f}%")
    print(f"Kappa Controller:   E[z] = {np.mean(z_kap):.4f} | Median = {np.median(z_kap):.4f} | P(z > 0.9*z_max) = {np.mean(z_kap > 0.45)*100:.1f}%")

    # --- Plot Histogram ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 0.5, 50)
    ax.hist(z_kap, bins=bins, alpha=0.6, color='#888888', label=r'Condition Number ($\kappa$)', density=True)
    ax.hist(z_eps, bins=bins, alpha=0.6, color='#DC143C', label=r'Spatial Subspace ($\epsilon$)', density=True)
    ax.set_title('Internal Diagnostic: Histogram of $\zeta$', fontweight='bold')
    ax.set_xlabel('Regularization Multiplier $\zeta$')
    ax.set_ylabel('Density')
    ax.legend(loc='upper center')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/diagnostic_loading_histogram.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved 'results/diagnostic_loading_histogram.png'")

if __name__ == "__main__":
    main()
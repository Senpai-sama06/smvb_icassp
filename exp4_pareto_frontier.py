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
from algos.cdr_mvdr import CDR_MVDR 
from algos.ulb import UniversalLinearBeamformer

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

def compute_mean_wng(weights):
    w_norm_sq = np.sum(np.abs(weights)**2, axis=-1)
    wng_db = -10 * np.log10(w_norm_sq + 1e-12)
    return np.mean(wng_db)

def main():
    print("--- RUNNING CLEANED EXP 4: PARETO FRONTIER ANALYSIS ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    M = 4 
    
    # 1. Simulate Severe Encroachment Scenario (Target 90°, Interferer 75°)
    print(">>> Simulating Encroachment (Target 90°, Interferer 75°) <<<")
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=20, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    mix, target, _, _ = simulator.simulate(
        n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False
    )
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)

    # Prepare Covariance Matrix
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
    F, T_frames = mix_stft.shape[1], mix_stft.shape[2]

    # 2. Sweep Static Loading to trace the Feasible Frontier
    static_zetas = np.logspace(-6, 0, 25)
    wng_static, sisdr_static = [], []
    
    ulb = UniversalLinearBeamformer(num_mics=M)
    
    print(">>> Tracing Static Pareto Curve...")
    for z in static_zetas:
        w = ulb.process(R_matrix, oracle_rtf, np.zeros((F, T_frames)), zeta=z)
        est = librosa.istft(ulb.apply_weights(mix_stft, w), hop_length=hop_length)
        wng_static.append(compute_mean_wng(w))
        sisdr_static.append(calculate_sisdr(target[0], est, mix[0]))

    # 3. Evaluate Dynamic Methods
    print(">>> Evaluating Proposed CDR and Binary Baseline...")
    cdr = CDR_MVDR(num_mics=M, ablation_mode=None)
    w_cdr = cdr.process(R_matrix, oracle_rtf)
    est_cdr = librosa.istft(cdr.apply_weights(mix_stft, w_cdr), hop_length=hop_length)
    wng_cdr = compute_mean_wng(w_cdr)
    sisdr_cdr = calculate_sisdr(target[0], est_cdr, mix[0])

    cdr_binary = CDR_MVDR(num_mics=M, ablation_mode='binary')
    w_bin = cdr_binary.process(R_matrix, oracle_rtf)
    est_bin = librosa.istft(cdr_binary.apply_weights(mix_stft, w_bin), hop_length=hop_length)
    wng_bin = compute_mean_wng(w_bin)
    sisdr_bin = calculate_sisdr(target[0], est_bin, mix[0])

    # --- Improved Plotting Aesthetics ---
    plt.figure(figsize=(10, 6.5))
    
    # Plot the bounding curve (Static Sweep)
    plt.plot(wng_static, sisdr_static, 'k--', linewidth=2.5, label='Static Loading (Feasible Frontier)', zorder=1)
    plt.scatter(wng_static, sisdr_static, c='lightgray', edgecolor='gray', s=50, zorder=2)
    
    # Highlight the Extremes
    plt.scatter(wng_static[0], sisdr_static[0], c='black', marker='o', s=180, zorder=5, label=f'Nominal MVDR ($\zeta=10^{-6}$)')
    plt.scatter(wng_static[-1], sisdr_static[-1], c='black', marker='s', s=180, zorder=5, label=f'Sledgehammer ($\zeta=1.0$)')
    
    # Plot Dynamic Methods
    plt.scatter(wng_bin, sisdr_bin, c='purple', marker='^', s=250, edgecolor='white', linewidth=1.5, zorder=6, label='Hard-Threshold (Binary)')
    plt.scatter(wng_cdr, sisdr_cdr, c='crimson', marker='D', s=250, edgecolor='white', linewidth=1.5, zorder=7, label='Proposed CDR (Autonomous)')
    
    plt.title('Pareto Frontier: Robustness vs. Spatial Selectivity (Encroachment)', fontsize=15, fontweight='bold')
    plt.xlabel('Array Robustness $\\rightarrow$ Mean WNG (dB)', fontsize=13)
    plt.ylabel('Spatial Selectivity $\\rightarrow$ SI-SDR Imp. (dB)', fontsize=13)
    
    # Fixed Annotations (Using offset points so they never go off-screen)
    plt.annotate('WNG Collapse\n(Noise Amplification)', 
                 xy=(wng_static[0], sisdr_static[0]), 
                 xytext=(30, 20), textcoords='offset points',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.annotate('Spatial Collapse\n(Interferer Leakage)', 
                 xy=(wng_static[-1], sisdr_static[-1]), 
                 xytext=(-120, -40), textcoords='offset points',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    # Highlight the Pareto Optimal Zone
    plt.axhspan(max(sisdr_static)-1.0, max(sisdr_static)+0.5, color='green', alpha=0.05, zorder=0)
    plt.text(np.mean(wng_static), max(sisdr_static)+0.1, 'Pareto Optimal Region', color='darkgreen', fontsize=12, fontweight='bold', ha='center')

    # Grid and Legend adjustments
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower center', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Dynamically pad the axes so nothing gets cut off
    plt.xlim(min(wng_static)-1.5, max(wng_static)+1.5)
    plt.ylim(min(sisdr_static)-1.5, max(sisdr_static)+1.5)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp4_pareto_frontier_v2.png', dpi=300)
    print("\n✅ Saved 'results/exp4_pareto_frontier_v2.png'")

if __name__ == "__main__":
    main()
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pathlib import Path
import time

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

def get_mean_wng(weights):
    # WNG = 10 * log10( 1 / ||w||^2 )
    w_norm_sq = np.sum(np.abs(weights)**2, axis=-1) + 1e-12
    wng_db = -10 * np.log10(w_norm_sq)
    return np.mean(wng_db)

def main():
    print("--- GENERATING FIG 7: ROBUSTNESS-SEPARATION FRONTIER ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    # Using a 65 degree encroachment (25 deg from target) to force a tough compromise
    print(">>> Simulating Encroachment (Target 90, Interferer 65) <<<")
    start_time = time.time()
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[65], save_outputs=False)
    mix = mix.T; target = target.T
    
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    if oracle_rtf.shape[0] == M and oracle_rtf.shape[1] != M:
        oracle_rtf = oracle_rtf.T
        
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
    
    # Precompute elements to speed up the static sweep
    trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / M
    _, evecs = np.linalg.eigh(R_matrix)
    u1 = evecs[..., -1]
    
    d_exp = oracle_rtf[:, np.newaxis, :]
    num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
    d_norm = np.linalg.norm(oracle_rtf, axis=-1)[:, np.newaxis]
    u1_norm = np.linalg.norm(u1, axis=-1)
    rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
    epsilon = 1.0 - (rho ** 2)

    ulb = UniversalLinearBeamformer(num_mics=M)
    F, T, _, _ = R_matrix.shape
    mu_inv = np.zeros((F, T), dtype=np.float32)

    # 1. Sweep Static Loading (The Frontier Curve)
    print(">>> Mapping Static Loading Frontier... <<<")
    zetas_static = np.logspace(-5, np.log10(0.5), 30)
    sweep_sisdr = []
    sweep_wng = []
    
    for z in zetas_static:
        zeta_tensor = np.full((F, T), z, dtype=np.float32) * trace_power
        w = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=zeta_tensor)
        est = librosa.istft(ulb.apply_weights(mix_stft, w), hop_length=hop_length)
        
        sweep_sisdr.append(calculate_sisdr(target[0], est, mix[0]))
        sweep_wng.append(get_mean_wng(w))

    # 2. Hard Threshold Policy
    print(">>> Evaluating Adaptive Policies... <<<")
    eps_0 = 1.0 / M
    z_hard = np.where(epsilon > eps_0, 0.5, 1e-5) * trace_power
    w_hard = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_hard)
    est_hard = librosa.istft(ulb.apply_weights(mix_stft, w_hard), hop_length=hop_length)
    hard_sisdr = calculate_sisdr(target[0], est_hard, mix[0])
    hard_wng = get_mean_wng(w_hard)
    
    # 3. Continuous (CDR) Policy
    z_cont = (1e-5 + (0.5 - 1e-5) / (1.0 + np.exp(-25.0 * (epsilon - eps_0)))) * trace_power
    w_cont = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_cont)
    est_cont = librosa.istft(ulb.apply_weights(mix_stft, w_cont), hop_length=hop_length)
    cont_sisdr = calculate_sisdr(target[0], est_cont, mix[0])
    cont_wng = get_mean_wng(w_cont)
    
    print(f"    Completed in {time.time() - start_time:.2f} seconds.")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the Frontier
    ax.plot(sweep_wng, sweep_sisdr, color='gray', linewidth=3, label='Static-loading sweep', zorder=2)
    
    # Plot Anchor Points from the sweep
    ax.scatter(sweep_wng[0], sweep_sisdr[0], color='forestgreen', marker='s', s=120, edgecolor='black', label='Nominal MVDR ($\\zeta_{min}$)', zorder=4)
    ax.scatter(sweep_wng[-1], sweep_sisdr[-1], color='darkred', marker='v', s=150, edgecolor='black', label='Fixed Heavy Load ($\\zeta_{max}$)', zorder=4)
    
    # Plot Adaptive Policies
    ax.scatter(hard_wng, hard_sisdr, color='#cccccc', marker='o', s=180, edgecolor='black', label=r'$\epsilon$ + Hard Threshold', zorder=5)
    ax.scatter(cont_wng, cont_sisdr, color='#4682b4', marker='*', s=400, edgecolor='black', label=r'$\epsilon$ + Continuous (CDR)', zorder=6)
    
    ax.set_title('Robustness–Separation Trade-off', fontweight='bold')
    ax.set_xlabel('Mean WNG (dB)', fontweight='bold')
    ax.set_ylabel('SI-SDRi (dB)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/fig7_pareto_frontier.png', dpi=1200, bbox_inches='tight')
    print("\n✅ Saved 'results/fig7_pareto_frontier.pdf'")

if __name__ == "__main__":
    main()
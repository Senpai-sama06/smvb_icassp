import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pathlib import Path
import time
import random

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
from algos.wng_mvdr import get_wng_constrained_loading

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
    w_norm_sq = np.sum(np.abs(weights)**2, axis=-1) + 1e-12
    wng_db = -10 * np.log10(w_norm_sq)
    return np.mean(wng_db)

def main():
    print("--- GENERATING FIG 8: SPATIAL ENCROACHMENT SWEEP (FINAL NOMENCLATURE) ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4
    
    angles = np.linspace(30, 85, 12)
    delta_thetas = 90 - angles
    
    # UPDATED LABELS: Perfect symmetry with the rest of the paper
    models = {
        'nominal': {'color': 'forestgreen', 'ls': '--', 'label': 'Nominal MVDR', 'lw': 2},
        'heavy': {'color': 'crimson', 'ls': ':', 'label': 'Fixed Heavy Load', 'lw': 2},
        'wng': {'color': 'darkorange', 'ls': '-.', 'label': 'WNG-MVDR', 'lw': 2},
        'hard': {'color': '#cccccc', 'ls': '-', 'label': 'HDR-MVDR', 'lw': 3},
        'cdr': {'color': '#4682b4', 'ls': '-', 'label': 'CDR-MVDR', 'lw': 3}
    }
    
    results = {m: {'sisdr': [], 'wng': []} for m in models}
    
    # We will average over 3 fixed seeds to ensure a smooth, statistically valid curve
    num_seeds = 3 
    
    for theta in angles:
        print(f">>> Simulating Interferer at {theta:.1f} deg (Delta = {90-theta:.1f} deg) <<<")
        start_time = time.time()
        
        seed_sisdr = {m: [] for m in models}
        seed_wng = {m: [] for m in models}
        
        for seed_val in range(42, 42 + num_seeds):
            # STRICT CONTROL: Force the simulator to pick the exact same speech/noise every time
            np.random.seed(seed_val)
            random.seed(seed_val)
            
            simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
            evaluator = Evaluator(ref_mic=0)
            
            mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[theta], save_outputs=False)
            mix = mix.T; target = target.T
            
            mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
            target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
            oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
            
            if oracle_rtf.shape[0] == M and oracle_rtf.shape[1] != M:
                oracle_rtf = oracle_rtf.T
                
            y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
            y_conj = y.conj().transpose(0, 1, 3, 2)
            R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
            
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
            eps_0 = 1.0 / M
            
            # 1. Nominal
            z_nom = np.full((F, T), 1e-5, dtype=np.float32) * trace_power
            w_nom = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_nom)
            seed_sisdr['nominal'].append(calculate_sisdr(target[0], librosa.istft(ulb.apply_weights(mix_stft, w_nom), hop_length=hop_length), mix[0]))
            seed_wng['nominal'].append(get_mean_wng(w_nom))
            
            # 2. Fixed Heavy
            z_hev = np.full((F, T), 0.5, dtype=np.float32) * trace_power
            w_hev = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_hev)
            seed_sisdr['heavy'].append(calculate_sisdr(target[0], librosa.istft(ulb.apply_weights(mix_stft, w_hev), hop_length=hop_length), mix[0]))
            seed_wng['heavy'].append(get_mean_wng(w_hev))
            
            # 3. WNG-Constrained Baseline 
            z_wng = get_wng_constrained_loading(R_matrix, oracle_rtf, gamma_db=0.0, z_init_max=0.5)
            w_wng = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_wng * trace_power)
            seed_sisdr['wng'].append(calculate_sisdr(target[0], librosa.istft(ulb.apply_weights(mix_stft, w_wng), hop_length=hop_length), mix[0]))
            seed_wng['wng'].append(get_mean_wng(w_wng))
            
            # 4. Hard (HDR-MVDR)
            z_hard = np.where(epsilon > eps_0, 0.5, 1e-5) * trace_power
            w_hard = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_hard)
            seed_sisdr['hard'].append(calculate_sisdr(target[0], librosa.istft(ulb.apply_weights(mix_stft, w_hard), hop_length=hop_length), mix[0]))
            seed_wng['hard'].append(get_mean_wng(w_hard))
            
            # 5. Continuous (CDR-MVDR)
            z_cont = (1e-5 + (0.5 - 1e-5) / (1.0 + np.exp(-25.0 * (epsilon - eps_0)))) * trace_power
            w_cont = ulb.process(R_matrix, oracle_rtf, mu_inv, zeta=z_cont)
            seed_sisdr['cdr'].append(calculate_sisdr(target[0], librosa.istft(ulb.apply_weights(mix_stft, w_cont), hop_length=hop_length), mix[0]))
            seed_wng['cdr'].append(get_mean_wng(w_cont))
            
        # Average the results for this angle
        for m in models:
            results[m]['sisdr'].append(np.mean(seed_sisdr[m]))
            results[m]['wng'].append(np.mean(seed_wng[m]))
            
        print(f"    Completed in {time.time() - start_time:.2f} seconds.")

    # --- Plotting ---
    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    
    for m_key, m_info in models.items():
        lw = m_info.get('lw', 2)
        axs[0].plot(delta_thetas, results[m_key]['sisdr'], color=m_info['color'], linestyle=m_info['ls'], linewidth=lw, label=m_info['label'])
        axs[1].plot(delta_thetas, results[m_key]['wng'], color=m_info['color'], linestyle=m_info['ls'], linewidth=lw, label=m_info['label'])

    axs[0].set_title('(a) Signal Separation (SI-SDRi)', fontweight='bold')
    axs[0].set_ylabel('dB')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    axs[1].set_title('(b) Array Robustness (Mean WNG)', fontweight='bold')
    axs[1].set_ylabel('dB')
    axs[1].set_xlabel(r'Angular Separation $\Delta\theta$ (°)', fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    fig.legend(handles=axs[0].get_lines(), loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=True, edgecolor='black', columnspacing=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/fig8_spatial_encroachment.pdf', dpi=300, bbox_inches='tight') 
    print("\n✅ Saved 'results/fig8_spatial_encroachment.pdf'")

if __name__ == "__main__":
    main()
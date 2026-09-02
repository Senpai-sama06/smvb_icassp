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
    'figure.titlesize': 13 # Slightly smaller to fit the acronym
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

def main():
    print("--- GENERATING FIG 6: 1/M vs ORACLE GAP (FINAL NOMENCLATURE) ---")
    fs, n_fft, hop_length = 16000, 1024, 256
    array_sizes = [2, 4, 8]
    
    # Sweep extended down to 0.005 to capture the true M=8 optimum
    thresholds = np.linspace(0.005, 0.70, 30)
    results = {M: [] for M in array_sizes}

    for M in array_sizes:
        print(f">>> Simulating {M}-Microphone Array Sweep <<<")
        start_time = time.time()
        
        simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs) 
        evaluator = Evaluator(ref_mic=0)
        
        i_angles = [65]
        if M == 4: i_angles = [65, 150, 240]
        elif M == 8: i_angles = [65, 120, 150, 180, 210, 240, 300]
            
        mix, target, _, _ = simulator.simulate(n=M-1, reverb=True, target_rt60=0.2, interferer_angles=i_angles, save_outputs=False)
        mix = mix.T; target = target.T
        mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        if oracle_rtf.shape[0] == M and oracle_rtf.shape[1] != M:
            oracle_rtf = oracle_rtf.T
            
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
        
        # --- PRECOMPUTE EVD AND EPSILON ONCE PER ARRAY (Massive Speedup) ---
        _, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        d_exp = oracle_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        d_norm = np.linalg.norm(oracle_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        
        trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / M
        
        ulb = UniversalLinearBeamformer(num_mics=M)
        F, T, _, _ = R_matrix.shape
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        
        # Now the threshold loop is just fast scalar math
        for thresh in thresholds:
            z_factor = 1e-5 + (0.5 - 1e-5) / (1.0 + np.exp(-25.0 * (epsilon - thresh)))
            zeta_tensor = z_factor * trace_power
            
            w_cdr = ulb.process(R_matrix, oracle_rtf, mu_inv_tensor, zeta=zeta_tensor)
            est_cdr = librosa.istft(ulb.apply_weights(mix_stft, w_cdr), hop_length=hop_length)
            results[M].append(calculate_sisdr(target[0], est_cdr, mix[0]))
            
        print(f"    Completed in {time.time() - start_time:.2f} seconds.")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {2: '#4682b4', 4: '#ffa500', 8: '#DC143C'}
    
    for M in array_sizes:
        ax.plot(thresholds, results[M], color=colors[M], linewidth=3, zorder=2)
        
        best_idx = np.argmax(results[M])
        oracle_thresh = thresholds[best_idx]
        oracle_sisdr = results[M][best_idx]
        
        proposed_thresh = 1.0 / M
        proposed_sisdr = np.interp(proposed_thresh, thresholds, results[M])
        delta_db = oracle_sisdr - proposed_sisdr
        
        # Plot Markers
        ax.scatter(proposed_thresh, proposed_sisdr, color=colors[M], marker='o', s=130, edgecolor='black', zorder=5)
        ax.scatter(oracle_thresh, oracle_sisdr, color='gold', marker='*', s=350, edgecolor='black', zorder=6)
        
        # Generate Label
        if delta_db > 0.05:
            ax.plot([oracle_thresh, proposed_thresh], [oracle_sisdr, proposed_sisdr], color='black', linestyle=':', linewidth=2, zorder=4)
            lbl = rf"$\Delta_{{rule}}={delta_db:.2f}$ dB"
        else:
            lbl = rf"$\Delta_{{rule}} \approx 0$ dB"
            
        # Manual Collision-Free Placements for Text Boxes
        if M == 2:
            anchor = ((oracle_thresh + proposed_thresh) / 2, (oracle_sisdr + proposed_sisdr) / 2)
            xytext, ha, va = (0, -15), 'center', 'top'
        elif M == 4:
            anchor = (proposed_thresh, proposed_sisdr)
            xytext, ha, va = (15, -15), 'left', 'top'
        elif M == 8:
            # Shifted to the left slightly to avoid the rising curve
            anchor = (proposed_thresh, proposed_sisdr)
            xytext, ha, va = (-15, 20), 'right', 'bottom' 
            
        ax.annotate(lbl, xy=anchor, xytext=xytext, textcoords='offset points',
                    ha=ha, va=va, fontsize=10, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor=colors[M], linewidth=1.5, boxstyle='round,pad=0.3'), zorder=7)

    # UPDATED TITLE
    ax.set_title(r'CDR-MVDR Oracle Gap: $1/M$ Transition Rule vs. Evaluated Best ($\epsilon_{0,\mathrm{best}}$)', fontweight='bold')
    ax.set_xlabel(r'Spatial Boundary Threshold $\epsilon_0$', fontweight='bold')
    ax.set_ylabel('SI-SDR Improvement (dB)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    import matplotlib.lines as mlines
    leg_lines = [mlines.Line2D([], [], color=colors[M], linewidth=3, label=f'$M={M}$ Array') for M in array_sizes]
    
    # UPDATED LEGEND
    leg_markers = [
        mlines.Line2D([], [], color='white', marker='*', markerfacecolor='gold', markeredgecolor='black', markersize=16, label=r'Evaluated Best ($\epsilon_{0,\mathrm{best}}$)'),
        mlines.Line2D([], [], color='white', marker='o', markerfacecolor='gray', markeredgecolor='black', markersize=10, label=r'CDR-MVDR ($1/M$ Heuristic)')
    ]
    ax.legend(handles=leg_lines + leg_markers, loc='lower right', framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/fig6_oracle_gap.pdf', dpi=300, bbox_inches='tight')
    print("\n✅ Saved 'results/fig6_oracle_gap.pdf'")

if __name__ == "__main__":
    main()
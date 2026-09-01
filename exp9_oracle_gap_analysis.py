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

class Oracle_Sweep_CDR:
    def __init__(self, num_mics=2, threshold=0.15, kappa=25.0, zeta_min=1e-5, zeta_max=0.5):
        self.M = num_mics
        self.epsilon_0 = threshold
        self.kappa = kappa
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)

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
        
        # MEAN TRACE NORMALIZATION 
        trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / self.M
        
        z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.kappa * (epsilon - self.epsilon_0)))
        zeta_tensor = z_factor * trace_power
        
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        return self.ulb.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)

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
    print("--- RUNNING EXP 9: ORACLE GAP ANALYSIS (MEAN-TRACE) ---")
    fs, n_fft, hop_length = 16000, 1024, 256
    array_sizes = [2, 4, 8]
    
    # We use a broad sweep to ensure we capture the absolute peak for all arrays
    thresholds = np.linspace(0.05, 0.70, 25)
    results = {M: [] for M in array_sizes}

    for M in array_sizes:
        print(f"\n>>> Simulating {M}-Microphone Array <<<")
        simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=20, fs=fs) 
        evaluator = Evaluator(ref_mic=0)
        
        # Moderate encroachment to force a clear robustness-selectivity trade-off
        i_angles = [60]
        if M == 4: i_angles = [60, 150, 240]
        elif M == 8: i_angles = [60, 120, 150, 180, 210, 240, 300]
            
        mix, target, _, _ = simulator.simulate(n=M-1, reverb=True, target_rt60=0.2, interferer_angles=i_angles, save_outputs=False)
        mix = mix.T; target = target.T
        mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
        
        for thresh in thresholds:
            cdr = Oracle_Sweep_CDR(num_mics=M, threshold=thresh)
            w_cdr = cdr.process(R_matrix, oracle_rtf)
            est_cdr = librosa.istft(cdr.apply_weights(mix_stft, w_cdr), hop_length=hop_length)
            results[M].append(calculate_sisdr(target[0], est_cdr, mix[0]))

    # --- Analysis & Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {2: 'steelblue', 4: 'orange', 8: 'crimson'}
    markers = {2: 'o', 4: 's', 8: 'D'}
    
    print("\n--- ORACLE GAP ANALYSIS REPORT ---")
    for M in array_sizes:
        # Find Oracle Peak
        best_idx = np.argmax(results[M])
        oracle_thresh = thresholds[best_idx]
        oracle_sisdr = results[M][best_idx]
        
        # Find 1/M Performance
        proposed_thresh = 1.0 / M
        proposed_sisdr = np.interp(proposed_thresh, thresholds, results[M])
        
        # Calculate Gaps
        delta_db = oracle_sisdr - proposed_sisdr
        
        print(f"M={M}: Oracle \epsilon_0* = {oracle_thresh:.3f} ({oracle_sisdr:.2f} dB) | "
              f"1/M = {proposed_thresh:.3f} ({proposed_sisdr:.2f} dB) | "
              f"Penalty = {delta_db:.2f} dB")
        
        # Plotting
        ax.plot(thresholds, results[M], color=colors[M], linewidth=2.5, label=f'M={M} Array')
        
        # Plot Oracle Star
        ax.scatter(oracle_thresh, oracle_sisdr, color='gold', marker='*', s=300, edgecolor='black', zorder=6)
        
        # Plot 1/M Point
        ax.scatter(proposed_thresh, proposed_sisdr, color=colors[M], marker=markers[M], s=150, edgecolor='black', zorder=5)
        
        # Draw line connecting them to show the gap visually
        ax.plot([oracle_thresh, proposed_thresh], [oracle_sisdr, proposed_sisdr], color='black', linestyle=':', linewidth=2)

    ax.set_title('Oracle Gap Analysis: $1/M$ Heuristic vs. Optimal Boundary ($\epsilon_0^\star$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Spatial Boundary Threshold $\epsilon_0$', fontsize=12)
    ax.set_ylabel('SI-SDR Improvement (dB)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    import matplotlib.lines as mlines
    line_legend = ax.legend(loc='lower right', fontsize=11)
    marker_legend1 = mlines.Line2D([], [], color='gold', marker='*', linestyle='None', markersize=12, markeredgecolor='black', label='Oracle Optimum ($\epsilon_0^\star$)')
    marker_legend2 = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Proposed 1/M Heuristic')
    ax.add_artist(line_legend)
    ax.legend(handles=[marker_legend1, marker_legend2], loc='lower center', fontsize=11)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp9_oracle_gap_analysis.png', dpi=300)
    print("\n✅ Saved 'results/exp9_oracle_gap_analysis.png'")

if __name__ == "__main__":
    main()
    
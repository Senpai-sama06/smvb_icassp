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

def compute_weight_jitter(weights):
    """
    Computes the mean squared frame-to-frame L2 norm difference of the weights.
    weights shape: (F, T, M)
    """
    delta_w = weights[:, 1:, :] - weights[:, :-1, :]
    l2_sq = np.sum(np.abs(delta_w)**2, axis=-1) 
    return np.mean(l2_sq)

def compute_zeta_jitter(zeta):
    """
    Computes the mean absolute frame-to-frame difference in the regularization parameter.
    zeta shape: (F, T)
    """
    delta_zeta = np.abs(zeta[:, 1:] - zeta[:, :-1])
    return np.mean(delta_zeta)

def main():
    print("--- RUNNING MISSING EXP 3: QUANTITATIVE TEMPORAL JITTER ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    M = 4 
    
    # Simulate an encroaching dynamic scenario with a close interferer angle
    print(">>> Simulating Dynamic Acoustic Scene <<<")
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    mix, target, _, _ = simulator.simulate(
        n=1, reverb=True, target_rt60=0.2, interferer_angles=[85], save_outputs=False
    )
    mix = mix.T; target = target.T
    
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    ablation_modes = ['nominal', 'binary', None] 
    labels = ['Nominal MVDR', 'Hard-Threshold\n(Binary Baseline)', 'Proposed CDR\n(Continuous)']
    
    w_jitter_results = []
    z_jitter_results = []
    
    for mode in ablation_modes:
        cdr = CDR_MVDR(num_mics=M, ablation_mode=mode)
        
        zeta_tensor = cdr.compute_zeta(R_matrix, oracle_rtf)
        weights = cdr.process(R_matrix, oracle_rtf)
        
        z_jitter = compute_zeta_jitter(zeta_tensor)
        w_jitter = compute_weight_jitter(weights)
        
        z_jitter_results.append(z_jitter)
        w_jitter_results.append(w_jitter)
        
        print(f"[{mode if mode else 'proposed'}]: Z-Jitter = {z_jitter:.6f}, W-Jitter = {w_jitter:.6f}")

    # --- Plotting the Quantitative Results ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['black', 'purple', 'crimson']
    
    axs[0].bar(labels, z_jitter_results, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    axs[0].set_yscale('log')
    axs[0].set_title('(a) Regularization Jitter ($|\Delta \zeta|$)', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Mean Frame-to-Frame Variance (Log Scale)', fontsize=12)
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.6)
    
    axs[1].bar(labels, w_jitter_results, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    axs[1].set_yscale('log')
    axs[1].set_title('(b) Filter Weight Jitter ($\|\Delta \mathbf{w}\|_2^2$)', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Mean Frame-to-Frame Variance (Log Scale)', fontsize=12)
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.suptitle('Quantitative Analysis of Temporal Filter Smoothness', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp3_temporal_jitter_quantification.png', dpi=300)
    print("\n✅ Saved 'results/exp3_temporal_jitter_quantification.png'")

if __name__ == "__main__":
    main()
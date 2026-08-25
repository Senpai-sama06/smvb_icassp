import os
import sys
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pathlib import Path
from scipy.signal import lfilter

# Ensure paths
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.diat import DIAT_Bf

def evaluate_bss(bss_function, X_bss, target, mix, evaluator, hop_length, **kwargs):
    """Helper to run a BSS algorithm and resolve permutation ambiguity."""
    try:
        Y_bss = bss_function(X_bss, **kwargs)
        if Y_bss.ndim == 3:
            Y_bss_stft = Y_bss.transpose(2, 0, 1)
        else:
            return None
            
        best_improvement = -999.0
        for i in range(Y_bss_stft.shape[0]):
            est_time = librosa.istft(Y_bss_stft[i], hop_length=hop_length)
            res = evaluator.evaluate_performance(target, mix, est_time)
            if res['improvement'] > best_improvement:
                best_improvement = res['improvement']
                
        return best_improvement
    except Exception as e:
        print(f"      [!] Failed to run {bss_function.__name__}: {e}")
        return None

def main():
    print("======================================================")
    print(" STEP 1: ENCROACHMENT ANGLE SWEEP (FIGURE GENERATOR)")
    print("======================================================")
    
    simulator = AcousticSceneSimulator(snr_target_db=25)
    evaluator = Evaluator(ref_mic=0)
    n_fft = 1024; hop_length = 256
    
    # Define the Encroachment Sweep
    angles = [30, 45, 60, 75, 80, 85, 88]
    
    # We use a try-except to safely fetch valid BSS algorithms
    bss_algorithms = []
    if hasattr(pra.bss, 'auxiva'): bss_algorithms.append(("AuxIVA", pra.bss.auxiva, {'n_iter': 30}))
    if hasattr(pra.bss, 'ilrma'): bss_algorithms.append(("ILRMA", pra.bss.ilrma, {'n_iter': 30, 'n_components': 2}))
    if hasattr(pra.bss, 'sparseauxiva'): bss_algorithms.append(("Sparse AuxIVA", pra.bss.sparseauxiva, {'n_iter': 30}))
    if hasattr(pra.bss, 'fastmnmf2'): bss_algorithms.append(("FastMNMF2", pra.bss.fastmnmf2, {'n_iter': 30, 'n_components': 2}))
    
    results_log = []

    for angle in angles:
        print(f"\n--- Simulating Encroachment Angle: {angle}° ---")
        
        mix, target, interferer, noise = simulator.simulate(
            n=1, reverb=True, target_rt60=0.2, interferer_angles=[angle], save_outputs=False
        )
        mix = mix.T; target = target.T
        
        mix_stft = np.stack([librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length), 
                             librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)])
        target_stft = np.stack([librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length), 
                                librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)])
        
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.7], [1, -0.7], np.matmul(y, y_conj), axis=1)

        X_bss = mix_stft.transpose(1, 2, 0)
        
        row_data = {'Angle': angle}
        
        # 1. Run BSS Baselines
        print("   -> Running Third-Party BSS...")
        for name, func, kwargs in bss_algorithms:
            score = evaluate_bss(func, X_bss, target, mix, evaluator, hop_length, **kwargs)
            if score is not None:
                row_data[f"Baseline: {name}"] = score
                
        # 2. Run DIAT-Bf Models
        print("   -> Running Proposed & Ablations...")
        diat_models = [
            ('Ablation: Static Sledgehammer', 'static'),
            ('Ablation: Binary Switch', 'binary'),
            ('Proposed: DIAT-Bf', None)
        ]
        
        for name, mode in diat_models:
            algo = DIAT_Bf(num_mics=2, ablation_mode=mode)
            weights = algo.process(R_matrix, oracle_rtf)
            est = librosa.istft(algo.apply_weights(mix_stft, weights), hop_length=hop_length)
            row_data[name] = evaluator.evaluate_performance(target, mix, est)['improvement']
            
        results_log.append(row_data)

    # Convert to DataFrame
    df = pd.DataFrame(results_log)
    df.set_index('Angle', inplace=True)
    
    # Save CSV
    df.to_csv("encroachment_results.csv")
    print("\n--- Data Saved to encroachment_results.csv ---")
    print(df.to_string())
    
    # Plotting
    print("\n--- Generating Figure ---")
    plt.figure(figsize=(10, 6))
    
    # Plot formatting
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for idx, column in enumerate(df.columns):
        linewidth = 2.5 if 'Proposed' in column else 1.5
        linestyle = '-' if 'Proposed' in column or 'Baseline' in column else '--'
        plt.plot(df.index, df[column], label=column, 
                 linewidth=linewidth, linestyle=linestyle, 
                 marker=markers[idx % len(markers)], markersize=6)

    plt.title('Performance vs. Angular Encroachment (Determined Regime)', fontsize=14, fontweight='bold')
    plt.xlabel('Interferer Angle (Degrees)', fontsize=12)
    plt.ylabel('SI-SDR Improvement (dB)', fontsize=12)
    plt.xticks(angles)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    
    # Save Plot
    plt.savefig('encroachment_plot.png', dpi=300)
    print("Plot successfully saved as 'encroachment_plot.png'.")

if __name__ == "__main__":
    main()
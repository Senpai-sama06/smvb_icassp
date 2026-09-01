import os
import sys
import random
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lfilter

from pesq import pesq
from pystoi import stoi

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.cdr_mvdr import CDR_MVDR 

def calculate_all_metrics(target, est, mix, fs=16000):
    min_len = min(len(target), len(est), len(mix))
    t, e, m = target[:min_len], est[:min_len], mix[:min_len]

    def si_sdr(ref, sig):
        ref = ref - np.mean(ref); sig = sig - np.mean(sig)
        alpha = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-12)
        target_scaled = alpha * ref
        noise_sig = sig - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise_sig, noise_sig) + 1e-12))
    
    sisdr_imp = si_sdr(t, e) - si_sdr(t, m)
    try: p_score = pesq(fs, t, e, 'wb')
    except: p_score = 1.0 
    s_score = stoi(t, e, fs, extended=False)

    return {'SI-SDR_Imp': sisdr_imp, 'PESQ': p_score, 'STOI': s_score}

def get_interferer_angles(M, sweep_angle):
    if M == 2: return [sweep_angle]
    elif M == 4: return [sweep_angle, 150, 240]
    elif M == 8: return [sweep_angle, 120, 150, 180, 210, 240, 300]
    return [sweep_angle]

def main():
    print("--- RUNNING EXPERIMENT 1: ENCROACHMENT ABLATION (M=2, 4, 8) ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    angles = [30, 45, 60, 75, 80, 85, 88]
    array_sizes = [2, 4, 8]
    
    ablation_modes = {
        'Nominal MVDR': 'nominal',
        'Sledgehammer': 'sledgehammer',
        'Compromise': 'compromise',
        'Linear Mapping': 'linear',
        'Binary Adaptive': 'binary',
        'Proposed CDR': None
    }
    
    master_results = {M: {'sisdr': [], 'pesq': [], 'stoi': []} for M in array_sizes}

    for M in array_sizes:
        print(f"\n>>> Simulating {M}-Microphone Array <<<")
        N = M - 1 
        
        simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=25, fs=fs) 
        evaluator = Evaluator(ref_mic=0)
        
        for angle in angles:
            print(f"  -> Encroaching Angle: {angle}°")
            random.seed(42); np.random.seed(42) 
            
            i_angles = get_interferer_angles(M, angle)
            
            mix, target, _, _ = simulator.simulate(
                n=N, reverb=True, target_rt60=0.2, interferer_angles=i_angles, save_outputs=False
            )
            
            mix = mix.T; target = target.T
            
            mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
            target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
            oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
            
            y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
            y_conj = y.conj().transpose(0, 1, 3, 2)
            alpha_ema = 0.98 
            R_matrix = lfilter([1 - alpha_ema], [1, -alpha_ema], np.matmul(y, y_conj), axis=1)
            
            row_sisdr, row_pesq, row_stoi = {'Angle': angle}, {'Angle': angle}, {'Angle': angle}
            
            for label, mode in ablation_modes.items():
                cdr = CDR_MVDR(num_mics=M, ablation_mode=mode)
                w = cdr.process(R_matrix, oracle_rtf)
                est = librosa.istft(cdr.apply_weights(mix_stft, w), hop_length=hop_length)
                metrics = calculate_all_metrics(target[0], est, mix[0], fs)
                
                row_sisdr[label] = metrics['SI-SDR_Imp']
                row_pesq[label] = metrics['PESQ']
                row_stoi[label] = metrics['STOI']

            master_results[M]['sisdr'].append(row_sisdr)
            master_results[M]['pesq'].append(row_pesq)
            master_results[M]['stoi'].append(row_stoi)

    # --- Plotting 3x3 Grid ---
    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    
    colors = {
        'Nominal MVDR': ('black', 'o', ':'),
        'Sledgehammer': ('gray', 's', '-.'),
        'Compromise': ('orange', 'v', '--'),
        'Linear Mapping': ('steelblue', 'x', '--'),
        'Binary Adaptive': ('purple', '^', '--'),
        'Proposed CDR': ('crimson', 'D', '-')
    }

    for col_idx, M in enumerate(array_sizes):
        df_sisdr = pd.DataFrame(master_results[M]['sisdr']).set_index('Angle')
        df_pesq = pd.DataFrame(master_results[M]['pesq']).set_index('Angle')
        df_stoi = pd.DataFrame(master_results[M]['stoi']).set_index('Angle')
        
        metrics = [
            (df_sisdr, f'SI-SDR Imp. (M={M})', axs[0, col_idx]),
            (df_pesq, f'PESQ (M={M})', axs[1, col_idx]),
            (df_stoi, f'STOI (M={M})', axs[2, col_idx])
        ]
        
        for df, title, ax in metrics:
            for algo in df.columns:
                c, m, ls = colors.get(algo, ('black', 'o', '-'))
                lw = 4.0 if algo == 'Proposed CDR' else 2.0
                ax.plot(df.index, df[algo], label=algo, linestyle=ls, color=c, marker=m, linewidth=lw)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(angles)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if col_idx == 0 and ax == axs[0, 0]: 
                ax.legend(loc="best", fontsize=10)
                
            if ax in axs[2, :]:
                ax.set_xlabel('Encroaching Interferer Angle (°)', fontsize=12)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp1_encroachment_ablation.png', dpi=300)
    print("\n✅ Saved 'results/exp1_encroachment_ablation.png'")

if __name__ == "__main__":
    main()
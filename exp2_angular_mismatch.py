import os
import sys
import random
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lfilter

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.cdr_mvdr import CDR_MVDR  

def calculate_sisdr_only(target, est, mix):
    min_len = min(len(target), len(est), len(mix))
    t, e, m = target[:min_len], est[:min_len], mix[:min_len]

    def si_sdr(ref, sig):
        ref = ref - np.mean(ref); sig = sig - np.mean(sig)
        alpha = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-12)
        target_scaled = alpha * ref
        noise_sig = sig - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise_sig, noise_sig) + 1e-12))
    
    return {'SI-SDR_Imp': si_sdr(t, e) - si_sdr(t, m)}

def main():
    print("--- RUNNING EXPERIMENT 2: STEERING VECTOR MISMATCH ABLATION ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    
    mismatch_errors = [0, 2, 4, 6, 8, 10, 12, 15]
    M = 4 
    N = M - 1
    
    true_target_angle = 90
    interferer_angles = [45, 135, 180] 
    
    results = []

    print(f"\n>>> Simulating {M}-Microphone Array with Static Scene <<<")
    
    # STRICTLY ANECHOIC: Isolates pure angular geometry
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=25, fs=fs) 
    evaluator = Evaluator(ref_mic=0)
    
    random.seed(42); np.random.seed(42) 
    mix, target, _, _ = simulator.simulate(
        n=N, reverb=False, interferer_angles=interferer_angles, save_outputs=False
    )
    
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    ablation_modes = {
        'Nominal MVDR': 'nominal',
        'Sledgehammer MVDR': 'sledgehammer',
        'Compromise MVDR': 'compromise',
        'Linear Mapping': 'linear',
        'Binary Adaptive': 'binary',
        'Proposed CDR-MVDR': None
    }
    
    for error in mismatch_errors:
        print(f"  -> Injecting Mismatch Error: {error}° (Assumed Target at {true_target_angle + error}°)")
        
        mismatch_angle = true_target_angle + error
        sim_mismatch = AcousticSceneSimulator(num_mics=M, fs=fs)
        room_mismatch = sim_mismatch.setup_room(reverb=False) 
        sim_mismatch.place_sources(room_mismatch, n=0) 
        
        mic_center = np.mean(sim_mismatch.mic_locs, axis=1)
        mismatch_pos = [
            mic_center[0] + sim_mismatch.radius * np.cos(mismatch_angle * np.pi / 180),
            mic_center[1] + sim_mismatch.radius * np.sin(mismatch_angle * np.pi / 180),
            1.5
        ]
        room_mismatch.sources[0].position = mismatch_pos
        room_mismatch.compute_rir()
        
        impulse = np.zeros(n_fft); impulse[0] = 1.0
        mismatch_stft = np.stack([librosa.stft(np.convolve(impulse, room_mismatch.rir[m][0]), n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        corrupted_rtf = evaluator.get_oracle_rtf(mismatch_stft, ref_mic=0)

        row_sisdr = {'Mismatch Error (°)': error}
        
        for label, mode in ablation_modes.items():
            cdr = CDR_MVDR(num_mics=M, ablation_mode=mode)
            w = cdr.process(R_matrix, corrupted_rtf)
            est = librosa.istft(cdr.apply_weights(mix_stft, w), hop_length=hop_length)
            row_sisdr[label] = calculate_sisdr_only(target[0], est, mix[0])['SI-SDR_Imp']
            
        results.append(row_sisdr)

    # --- Plotting ---
    df = pd.DataFrame(results).set_index('Mismatch Error (°)')
    
    plt.figure(figsize=(10, 6))
    colors = {
        'Nominal MVDR': ('black', 'o', ':'),
        'Sledgehammer MVDR': ('gray', 's', '-.'),
        'Compromise MVDR': ('orange', 'v', '--'),
        'Linear Mapping': ('steelblue', 'x', '--'),
        'Binary Adaptive': ('purple', '^', '--'),
        'Proposed CDR-MVDR': ('crimson', 'D', '-')
    }
    
    for algo in df.columns:
        c, m, ls = colors[algo]
        lw = 4.0 if algo == 'Proposed CDR-MVDR' else 2.0
        plt.plot(df.index, df[algo], label=algo, linestyle=ls, color=c, marker=m, linewidth=lw)

    plt.title('Ablation Study: Robustness to Steering Vector Mismatch (M=4)', fontsize=14, fontweight='bold')
    plt.xlabel('Angular Mismatch Error (°)', fontsize=12)
    plt.ylabel('SI-SDR Improvement (dB)', fontsize=12)
    plt.xticks(mismatch_errors)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp2_steering_mismatch_ablation.png', dpi=300)
    print("\n✅ Saved 'results/exp2_steering_mismatch_ablation.png'")

if __name__ == "__main__":
    main()
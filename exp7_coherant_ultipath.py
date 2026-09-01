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
from algos.cdr_mvdr import CDR_MVDR  # Our auto-calibrating subspace leakage class

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
    print("--- RUNNING EXPERIMENT 7: COHERENT MULTIPATH (REVERBERATION) ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    
    rt60_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    M = 4 
    N = 3 # 3 interferers
    
    # Safe spatial setup: No encroachment, no steering mismatch
    true_target_angle = 90
    interferer_angles = [45, 135, 180] 
    
    results = []
    evaluator = Evaluator(ref_mic=0)

    # 1. Extract the Anechoic (Direct-Path) RTF Once
    # In real systems, the assumed steering vector is always the direct path.
    print(">>> Extracting Anechoic Direct-Path RTF (90°) <<<")
    sim_anechoic = AcousticSceneSimulator(num_mics=M, fs=fs)
    room_anechoic = sim_anechoic.setup_room(reverb=False)
    sim_anechoic.place_sources(room_anechoic, n=0) # Target only
    
    mic_center = np.mean(sim_anechoic.mic_locs, axis=1)
    target_pos = [
        mic_center[0] + sim_anechoic.radius * np.cos(true_target_angle * np.pi / 180),
        mic_center[1] + sim_anechoic.radius * np.sin(true_target_angle * np.pi / 180),
        1.5
    ]
    room_anechoic.sources[0].position = target_pos
    room_anechoic.compute_rir()
    
    impulse = np.zeros(n_fft); impulse[0] = 1.0
    direct_stft = np.stack([librosa.stft(np.convolve(impulse, room_anechoic.rir[m][0]), n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    direct_rtf = evaluator.get_oracle_rtf(direct_stft, ref_mic=0)

    ablation_modes = {
        'Nominal MVDR': 'nominal',
        'Sledgehammer': 'sledgehammer',
        'Compromise': 'compromise',
        'Linear Mapping': 'linear',
        'Binary Adaptive': 'binary',
        'Proposed CDR': None
    }
    
    # 2. Sweep Room Reverberation
    for rt60 in rt60_values:
        print(f"\n  -> Simulating Room Reverberation: RT60 = {rt60}s")
        reverb_flag = rt60 > 0.0
        
        simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs) 
        random.seed(42); np.random.seed(42) 
        
        mix, target, _, _ = simulator.simulate(
            n=N, reverb=reverb_flag, target_rt60=rt60 if reverb_flag else 0.2, 
            interferer_angles=interferer_angles, save_outputs=False
        )
        
        mix = mix.T; target = target.T
        mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

        row_sisdr = {'RT60 (s)': rt60}
        
        # 3. Process the reverberant mixture using the direct-path RTF
        for label, mode in ablation_modes.items():
            cdr = CDR_MVDR(num_mics=M, ablation_mode=mode)
            w = cdr.process(R_matrix, direct_rtf)
            est = librosa.istft(cdr.apply_weights(mix_stft, w), hop_length=hop_length)
            row_sisdr[label] = calculate_sisdr_only(target[0], est, mix[0])['SI-SDR_Imp']
            
        results.append(row_sisdr)

    # --- Print and Plot the Multipath Table ---
    df = pd.DataFrame(results).set_index('RT60 (s)')
    
    print("\n" + "="*80)
    print("EXPERIMENT 7: ROBUSTNESS TO COHERENT MULTIPATH (SI-SDR Improvement in dB)")
    print("="*80)
    print(df.round(2).to_markdown())
    print("="*80)
    
    plt.figure(figsize=(10, 6))
    colors = {
        'Nominal MVDR': ('black', 'o', ':'),
        'Sledgehammer': ('gray', 's', '-.'),
        'Compromise': ('orange', 'v', '--'),
        'Linear Mapping': ('steelblue', 'x', '--'),
        'Binary Adaptive': ('purple', '^', '--'),
        'Proposed CDR': ('crimson', 'D', '-')
    }
    
    for algo in df.columns:
        c, m, ls = colors[algo]
        lw = 4.0 if algo == 'Proposed CDR' else 2.0
        plt.plot(df.index, df[algo], label=algo, linestyle=ls, color=c, marker=m, linewidth=lw)

    plt.title('Robustness to Coherent Multipath / Reverberation (M=4)', fontsize=14, fontweight='bold')
    plt.xlabel('Room Reverberation Time - RT60 (seconds)', fontsize=12)
    plt.ylabel('SI-SDR Improvement (dB)', fontsize=12)
    plt.xticks(rt60_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp7_coherent_multipath.png', dpi=300)
    print("\n✅ Saved 'results/exp7_coherent_multipath.png'")

if __name__ == "__main__":
    main()
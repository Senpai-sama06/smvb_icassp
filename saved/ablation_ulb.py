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

def main():
    print("--- RUNNING STEP 1: ULB INTERNAL ABLATION (TRACE-NORMALIZED) ---")
    fs = 16000
    n_fft = 1024; hop_length = 256
    M = 2
    
    test_angles = [45, 88]
    # Sweep epsilon logarithmically from 10^-5 to 10^1
    epsilons = np.logspace(-5, 1, 15)
    results = {45: [], 88: []}
    
    simulator = AcousticSceneSimulator(snr_target_db=10, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    ulb = UniversalLinearBeamformer(num_mics=M)

    for angle in test_angles:
        print(f"\nSimulating Acoustic Scene for Angle: {angle}°")
        np.random.seed(42)
        
        mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[angle], save_outputs=False)
        mix = mix.T; target = target.T
        
        mix_stft = np.stack([librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length), 
                             librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)])
        target_stft = np.stack([librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length), 
                                librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)])
        
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        # Calculate raw Sample Covariance Matrix
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        
        # FIX: Using alpha = 0.95 for stable covariance tracking to prevent self-cancellation
        R_raw = lfilter([1 - 0.95], [1, -0.95], np.matmul(y, y_conj), axis=1) 
        F, T = mix_stft.shape[1], mix_stft.shape[2]
        
        # Pre-compute the Trace for scale-invariant loading
        trace_R = np.trace(R_raw, axis1=2, axis2=3) # Shape: (F, T)
        I = np.eye(M).reshape(1, 1, M, M)

        print(f"Sweeping Trace-Normalized Epsilon for {angle}°...")
        for eps in epsilons:
            # FIX: Apply Trace-Normalized Diagonal Loading
            R_reg = R_raw + eps * (trace_R[..., np.newaxis, np.newaxis] / M) * I
            
            # Feed the pre-regularized matrix to the ULB (setting internal zeta near 0)
            w_ulb = ulb.process(R_reg, oracle_rtf, np.zeros((F,T)), zeta=1e-12)
            est_stft = ulb.apply_weights(mix_stft, w_ulb)
            est_time = librosa.istft(est_stft, hop_length=hop_length)
            
            res = evaluator.evaluate_performance(target[0], mix[0], est_time)
            results[angle].append(res['improvement'])

    # --- PLOTTING ---simulator = AcousticSceneSimulator(snr_target_db=10, fs=fs) # Lower SNR to simulate cheap mics
    plt.figure(figsize=(9, 6))
    
    plt.plot(epsilons, results[45], marker='o', linestyle='-', color='teal', linewidth=2.5, label='45° (Safe Separation)')
    plt.plot(epsilons, results[88], marker='s', linestyle='-', color='crimson', linewidth=2.5, label='88° (Severe Encroachment)')
    
    plt.xscale('log')
    plt.title('Vanilla ULB Ablation: The Failure of Static Parameters', fontsize=14, fontweight='bold')
    plt.xlabel('Trace-Normalized Diagonal Loading ($\\epsilon$)', fontsize=12)
    plt.ylabel('SI-SDR Improvement (dB)', fontsize=12)
    
    # Visual aides
    plt.axvline(x=1e-3, color='gray', linestyle='--', alpha=0.7, label='Typical Static Tuning')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig('ulb_ablation_proof.png', dpi=300)
    print("\nSaved 'ulb_ablation_proof.png'.")

if __name__ == "__main__":
    main()
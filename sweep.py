import os
import sys
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from scipy.signal import lfilter
from itertools import product
from tqdm import tqdm

# Ensure paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(CURRENT_DIR), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from smvb_icassp.util.simulator import AcousticSceneSimulator
from smvb_icassp.util.evaluator import Evaluator
from smvb_icassp.algos.ulb import UniversalLinearBeamformer

def main():
    # 25 dB SNR to represent real-world hardware
    simulator = AcousticSceneSimulator(snr_target_db=25)
    evaluator = Evaluator(ref_mic=0)
    ulb = UniversalLinearBeamformer(num_mics=2)
    
    n_fft = 1024
    hop_length = 256

    # 1. Physical States (The raw causes of rho and eta)
    # 0 to 70 in steps of 10. Then 75, 80, 82, 84, 86, 88 for encroachment mapping.
    angles = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 82, 84, 86, 88] 
    
    # Low, medium, and high reverb
    rt60s = [0.0, 0.2, 0.4, 0.6] 
    
    # 2. Mathematical States (Slightly higher resolution to see the curves)
    mu_invs = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2]
    zetas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    results = []

    print("--- Starting Raw Heuristic Parameter Sweep ---")
    
    # Loop over every physical scene
    for angle, rt60 in tqdm(
    product(angles, rt60s),
    total=len(angles) * len(rt60s),
    desc="Scene sweep",
    unit="scene"):
        
        # Generate raw scene
        mix, target, interferer, noise = simulator.simulate(
            n=1, reverb=(rt60 > 0), target_rt60=rt60, 
            interferer_angles=[angle], save_outputs=False
        )
        
        mix_stft = np.stack([
            librosa.stft(mix.T[0], n_fft=n_fft, hop_length=hop_length),
            librosa.stft(mix.T[1], n_fft=n_fft, hop_length=hop_length)
        ])
        target_stft = np.stack([
            librosa.stft(target.T[0], n_fft=n_fft, hop_length=hop_length),
            librosa.stft(target.T[1], n_fft=n_fft, hop_length=hop_length)
        ])
        
        F, T = mix_stft.shape[1], mix_stft.shape[2]
        
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0).T
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_inst = np.matmul(y, y_conj)
        
        # Basic smoothing to make covariance matrix usable
        R_matrix = lfilter([1 - 0.7], [1, -0.7], R_inst, axis=1)
        
        # Grid Search all math parameters against this single physical scene
        for mu_val, zeta_val in product(mu_invs, zetas):
            
            mu_inv_tensor = np.full((F, T), mu_val, dtype=np.float32)
            
            # Run ULB
            weights = ulb.process(R_matrix, oracle_rtf, mu_inv_tensor, zeta=zeta_val)
            enhanced_stft = ulb.apply_weights(mix_stft, weights)
            enhanced_time = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            res = evaluator.evaluate_performance(target.T, mix.T, enhanced_time)
            
            # Record EVERYTHING. No filters. No "only best".
            results.append({
                'Angle': angle,
                'RT60': rt60,
                'Mu_Inv': mu_val,
                'Zeta': zeta_val,
                'SI_SDR_Imp': round(res['improvement'], 3)
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("raw_heuristic_sweep.csv", index=False)
    print("\nSweep Complete! Results saved to raw_heuristic_sweep.csv")

if __name__ == "__main__":
    main()
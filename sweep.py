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

def calculate_physical_metrics(R_matrix, oracle_rtf):
    """Calculates average Collinearity (rho) and Anisotropy (eta) for the scene."""
    F, T, M, M2 = R_matrix.shape
    rho_list, eta_list = [], []
    
    for f in range(F):
        d_f = oracle_rtf[f, :]
        d_norm = np.linalg.norm(d_f) + 1e-12
        
        # Average covariance over time for this frequency
        R_f = np.mean(R_matrix[f, :, :, :], axis=0)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R_f)
        
        # Anisotropy (eta) -> lambda_max / lambda_min
        lambda_max = np.abs(eigenvalues[-1])
        lambda_min = np.abs(eigenvalues[0])
        eta_list.append(lambda_max / (lambda_min + 1e-12))
        
        # Collinearity (rho) -> cosine similarity with dominant eigenvector
        u_1 = eigenvectors[:, -1]
        u_norm = np.linalg.norm(u_1) + 1e-12
        rho = np.abs(np.vdot(d_f, u_1)) / (d_norm * u_norm)
        rho_list.append(rho)
        
    return np.mean(rho_list), np.mean(eta_list)

def main():
    simulator = AcousticSceneSimulator(snr_target_db=25)
    evaluator = Evaluator(ref_mic=0)
    ulb = UniversalLinearBeamformer(num_mics=2)
    
    n_fft = 1024
    hop_length = 256

    # 1. Define the Physics Sweep (Varying rho and eta)
    angles_to_test = [30, 60, 80, 88]          # Drives rho (Encroachment)
    rt60s_to_test = [0.0, 0.2, 0.4, 0.6]       # Drives eta (Diffuseness/Overload)
    
    # 2. Define the Math Sweep (Varying ULB parameters)
    mu_inv_vals = [0.0, 0.1, 0.5, 1.0]         # From MVDR to Wiener
    zeta_vals = [1e-6, 1e-4, 1e-2, 1e-1]       # Diagonal regularization (Interference Penalty)
    
    results_log = []

    print("--- Starting Oracle Parameter Sweep ---")
    
    for angle, rt60 in tqdm(product(angles_to_test, rt60s_to_test), total=len(angles_to_test) * len(rt60s_to_test),
    desc="Physics sweep"):
        # print(f"\nSimulating Room: Angle={angle}°, RT60={rt60}s")
        
        # Generate Scene
        mix, target, interferer, noise = simulator.simulate(
            n=1, reverb=(rt60>0), target_rt60=rt60, 
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
        
        # Setup Matrices
        target_stft_transposed = target_stft.transpose(1, 0, 2)
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0).T # (F, M)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_inst = np.matmul(y, y_conj)
        R_matrix = lfilter([1 - 0.9], [1, -0.9], R_inst, axis=1)
        
        # Extract Physical Metrics
        avg_rho, avg_eta = calculate_physical_metrics(R_matrix, oracle_rtf)
        
        best_sisdr = -999.0
        best_params = (0.0, 1e-6)
        
        # Grid Search the Parameters
        for mu_val, zeta_val in (product(mu_inv_vals, zeta_vals)):
            mu_inv_tensor = np.full((F, T), mu_val, dtype=np.float32)
            
            weights = ulb.process(R_matrix, oracle_rtf, mu_inv_tensor, zeta=zeta_val)

            enhanced_stft = ulb.apply_weights(mix_stft, weights)
            enhanced_time = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            res = evaluator.evaluate_performance(target.T, mix.T, enhanced_time)
            
            if res['improvement'] > best_sisdr:
                best_sisdr = res['improvement']
                best_params = (mu_val, zeta_val)
                
        # print(f"  Physics -> Rho: {avg_rho:.3f}, Eta: {avg_eta:.3f}")
        # print(f"  Optimal Math -> mu_inv: {best_params[0]}, zeta: {best_params[1]}")
        # print(f"  Max SI-SDR Imp: {best_sisdr:+.2f} dB")
        
        results_log.append({
            'Angle': angle, 'RT60': rt60,
            'Rho_avg': avg_rho, 'Eta_avg': avg_eta,
            'Optimal_Mu_Inv': best_params[0],
            'Optimal_Zeta': best_params[1],
            'Max_SI_SDR': best_sisdr
        })

    # Save to CSV for analysis
    df = pd.DataFrame(results_log)
    df.to_csv("ulb_sweep_results.csv", index=False)
    print("\nSweep Complete! Results saved to ulb_sweep_results.csv")

if __name__ == "__main__":
    main()
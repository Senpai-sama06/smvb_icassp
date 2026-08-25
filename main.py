import os
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import random
from scipy.signal import lfilter

# Set deterministic seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(CURRENT_DIR), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from smvb_icassp.util.simulator import AcousticSceneSimulator
from smvb_icassp.util.evaluator import Evaluator
from smvb_icassp.algos.ulb import UniversalLinearBeamformer  # Import your new class

def main():
    
    print("--- 1. Generating Acoustic Scene (Validation Mode) ---")
    
    # Initialize simulator with realistic 25 dB sensor noise
    simulator = AcousticSceneSimulator(snr_target_db=25) 
    
    # TESTCASE: N=1, Encroachment Trap, Low Reverb
    mix, target, interferer, noise = simulator.simulate(
        n=1, 
        reverb=True, 
        target_rt60=0.2, 
        interferer_angles=[30], 
        save_outputs=True
    )
    
    mix = mix.T
    target = target.T
    
    print("\n--- 2. STFT Transformation ---")
    n_fft = 1024
    hop_length = 256
    
    mix_stft = np.stack([
        librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length),
        librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)
    ])
    
    target_stft = np.stack([
        librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length),
        librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)
    ])
    
    print("--- 3. Estimating Oracle RTF & Covariance ---")
    evaluator = Evaluator(ref_mic=0)
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    # Calculate empirical spatial covariance matrix (R_yy)
    M, F, T = mix_stft.shape
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]       # (F, T, M, 1)
    y_conj = y.conj().transpose(0, 1, 3, 2)                # (F, T, 1, M)
    R_inst = np.matmul(y, y_conj)                          # Instantaneous (F, T, M, M)
    
    # Smooth over time using simple EMA (alpha = 0.9)
    R_matrix = lfilter([1 - 0.9], [1, -0.9], R_inst, axis=1) 
    
    print("--- 4. Running ULB (Forced MVDR State) ---")
    ulb = UniversalLinearBeamformer(num_mics=2)
    
    # Force mu_inv to 0.0 (Strict MVDR / Distortionless mode)
    mu_inv = np.zeros((F, T), dtype=np.float32) 
    
    # Calculate weights and apply them
    weights = ulb.process(R_matrix, oracle_rtf, mu_inv)
    enhanced_stft = ulb.apply_weights(mix_stft, weights)
    
    print("--- 5. Inverse STFT ---")
    enhanced_time = librosa.istft(enhanced_stft, hop_length=hop_length)
    
    print("--- 6. Evaluation ---")
    results = evaluator.evaluate_performance(target, mix, enhanced_time)
    
    print(f"\nVALIDATION RESULTS:")
    print(f"  Input SI-SDR (Mic 1) : {results['sisdr_in']:.2f} dB")
    print(f"  Output SI-SDR (ULB)  : {results['sisdr_out']:.2f} dB")
    print(f"  Improvement          : {results['improvement']:+.2f} dB")

if __name__ == "__main__":
    main()
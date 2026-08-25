import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import lfilter

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.v_smvb import SMVB
from algos.ulb import UniversalLinearBeamformer
from algos.diat import DIAT_Bf  # Our new algorithm!

def main():
    print("--- 1. Generating Acoustic Scene (The 88-Degree Encroachment Trap) ---")
    simulator = AcousticSceneSimulator(snr_target_db=25)
    
    mix, target, interferer, noise = simulator.simulate(
        n=1, reverb=True, target_rt60=0.2, interferer_angles=[88], save_outputs=False
    )
    mix = mix.T; target = target.T
    
    n_fft = 1024; hop_length = 256
    mix_stft = np.stack([librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)])
    target_stft = np.stack([librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)])
    
    F, T = mix_stft.shape[1], mix_stft.shape[2]
    
    evaluator = Evaluator(ref_mic=0)
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    # Calculate Instantaneous Covariance Matrix
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_inst = np.matmul(y, y_conj)
    R_matrix = lfilter([1 - 0.7], [1, -0.7], R_inst, axis=1)

    print("\n--- 2. Running Algorithms ---")
    
    # --- MODEL 1: STATIC MVDR (The Baseline Failure) ---
    print("Running Static MVDR...")
    ulb = UniversalLinearBeamformer(num_mics=2)
    weights_mvdr = ulb.process(R_matrix, oracle_rtf, np.zeros((F,T)), zeta=1e-6)
    est_mvdr = librosa.istft(ulb.apply_weights(mix_stft, weights_mvdr), hop_length=hop_length)
    
    # --- MODEL 2: OLD SMVB (The Hard-Switching Flaw) ---
    print("Running Old SMVB...")
    smvb = SMVB(num_mics=2)
    est_smvb = librosa.istft(smvb.process(mix_stft, oracle_rtf), hop_length=hop_length)
    
    # --- MODEL 3: NEW DIAT-Bf (The Continuous Solution) ---
    print("Running New DIAT-Bf...")
    diat = DIAT_Bf(num_mics=2)
    weights_diat = diat.process(R_matrix, oracle_rtf)
    est_diat = librosa.istft(diat.apply_weights(mix_stft, weights_diat), hop_length=hop_length)

    print("\n--- 3. Final SI-SDR Results (The 88-Degree Trap) ---")
    res_mvdr = evaluator.evaluate_performance(target, mix, est_mvdr)
    res_smvb = evaluator.evaluate_performance(target, mix, est_smvb)
    res_diat = evaluator.evaluate_performance(target, mix, est_diat)
    
    print(f"Input SI-SDR      : {res_mvdr['sisdr_in']:.2f} dB")
    print("-" * 40)
    print(f"1. Static MVDR    : {res_mvdr['improvement']:+.2f} dB (Algorithmic Suicide)")
    print(f"2. Old SMVB       : {res_smvb['improvement']:+.2f} dB (Binary Chatter)")
    print(f"3. New DIAT-Bf    : {res_diat['improvement']:+.2f} dB (Autonomous Coordination)")
    
    # Save the files so you can HEAR the chatter vs the clean DIAT output
    min_len = min(len(target[0]), len(est_mvdr))
    sf.write("1_out_mvdr.wav", est_mvdr[:min_len], simulator.fs)
    sf.write("2_out_smvb.wav", est_smvb[:min_len], simulator.fs)
    sf.write("3_out_diat.wav", est_diat[:min_len], simulator.fs)
    print("\nSaved audio outputs to disk. Put on your headphones and listen to the difference between 2_out_smvb and 3_out_diat!")

if __name__ == "__main__":
    main()
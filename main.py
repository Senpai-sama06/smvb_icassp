import os
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

import random
import numpy as np

from algos import smvb

# Set deterministic seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Ensure parent and package directories are on sys.path for direct execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(CURRENT_DIR), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from smvb_icassp.util.simulator import AcousticSceneSimulator
    from smvb_icassp.util.evaluator import Evaluator
    from smvb_icassp.algos.v_smvb import SMVB
except ImportError:
    from util.simulator import AcousticSceneSimulator
    from util.evaluator import Evaluator
    from algos.v_smvb import SMVB


def main():
    print("--- 1. Generating Acoustic Scene ---")
    # Initialize simulator and generate 2-interferer overloaded scene in a reverberant room
    simulator = AcousticSceneSimulator()
    mix, target, interferer, noise = simulator.simulate(
        n=2, reverb=True, target_rt60=0.3, save_outputs=False
    )
    
    # Transpose to shape (Channels, Samples)
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
    
    print("--- 3. Estimating Oracle RTF ---")
    # This replicates the SPCOM Oracle constraint using Evaluator
    evaluator = Evaluator(ref_mic=0)
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    print("--- 4. Running SMVB Baseline ---")
    beamformer = SMVB(num_mics=2)
    enhanced_stft = beamformer.process(mix_stft, oracle_rtf)
    
    print("--- 5. Inverse STFT ---")
    enhanced_time = librosa.istft(enhanced_stft, hop_length=hop_length)
    
    print("--- 6. Evaluation ---")
    # Calculate SI-SDR metrics against the reverberant spatial image at Mic 1
    results = evaluator.evaluate_performance(target, mix, enhanced_time)
    
    print(f"\nRESULTS:")
    print(f"  Input SI-SDR (Mic 1) : {results['sisdr_in']:.2f} dB")
    print(f"  Output SI-SDR (SMVB) : {results['sisdr_out']:.2f} dB")
    print(f"  Improvement          : {results['improvement']:+.2f} dB")
    
    # Align lengths due to iSTFT padding and save output
    min_len = min(len(target[0]), len(enhanced_time))
    enhanced_signal = enhanced_time[:min_len]
    sf.write("output_smvb.wav", enhanced_signal, simulator.fs)
    print("\nSaved enhanced audio to 'output_smvb.wav'")


if __name__ == "__main__":
    main()
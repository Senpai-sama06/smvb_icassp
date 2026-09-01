import os
import sys
import json
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Ensure proper importing from the project structure
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import your modular utilities and algorithms
from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
# from algos.smvb import SMVB  
from algos.v_smvb import SMVB 

# --- 1. Define the 5 Validation Edge-Cases ---
# Target is always mathematically placed at 90 degrees by the simulator.
TEST_CASES = {
    "1_Standard_Overload": {
        "n": 2, "reverb": True, "target_rt60": 0.3, "angles": [30, 150], "seed": 42,
        "desc": "Standard 2-interferer sparsity test."
    },
    "2_Collinear_Encroachment": {
        "n": 1, "reverb": True, "target_rt60": 0.3, "angles": [85], "seed": 101,
        "desc": "Interferer at 85° forces the collinearity metric to spike."
    },
    "3_Diffuse_Washout": {
        "n": 2, "reverb": True, "target_rt60": 0.6, "angles": [45, 135], "seed": 202,
        "desc": "High RT60 to test covariance robustness against diffuse tails."
    },
    "4_Extreme_Starvation": {
        "n": 3, "reverb": True, "target_rt60": 0.3, "angles": [30, 150, 270], "seed": 303,
        "desc": "N=3 on M=2 array to force constant spatial entropy overload."
    },
    "5_General_Stress": {
        "n": 2, "reverb": True, "target_rt60": 0.4, "angles": [70, 110], "seed": 404,
        "desc": "Tight spatial clustering with moderate reverberation."
    }
}

def generate_testcases(base_dir="testcases"):
    """Generates and saves the audio files for the 5 test cases if missing."""
    os.makedirs(base_dir, exist_ok=True)
    simulator = AcousticSceneSimulator()

    for name, cfg in TEST_CASES.items():
        case_dir = os.path.join(base_dir, name)
        os.makedirs(case_dir, exist_ok=True)
        
        mix_path = os.path.join(case_dir, "mixture.wav")
        target_path = os.path.join(case_dir, "target.wav")
        
        # Only simulate if the files don't already exist
        if not os.path.exists(mix_path):
            print(f"Generating Audio for: {name}...")
            
            # Lock seeds for absolute determinism
            np.random.seed(cfg["seed"])
            random.seed(cfg["seed"])
            
            mix, target, interf, noise = simulator.simulate(
                n=cfg["n"], 
                reverb=cfg["reverb"], 
                target_rt60=cfg["target_rt60"], 
                interferer_angles=cfg["angles"], 
                save_outputs=False
            )
            
            # Save audio directly to the specific testcase folder
            sf.write(mix_path, mix, simulator.fs)
            sf.write(target_path, target, simulator.fs)
            
            # Save configuration metadata
            with open(os.path.join(case_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=4)
        else:
            print(f"Found existing data for: {name}")

def run_evaluations(base_dir="testcases"):
    """Loads the test cases and runs all registered algorithms."""
    # Register your algorithms here
    algos = {
        "SMVB_Baseline": SMVB(num_mics=2),
        # "DIAT_Bf_v1": DIAT_Bf(...),  <-- We will drop DIAT-Bf here soon
    }
    
    evaluator = Evaluator(ref_mic=0)
    n_fft, hop_length = 1024, 256
    
    print("\n" + "="*70)
    print(f"{'Test Case':<25} | {'Algorithm':<15} | {'SI-SDR In'} | {'SI-SDR Out'} | {'Improvement'}")
    print("="*70)

    for name in TEST_CASES.keys():
        case_dir = os.path.join(base_dir, name)
        
        # Load audio shapes as (M, Samples)
        mix, sr = librosa.load(os.path.join(case_dir, "mixture.wav"), sr=None, mono=False)
        target, _ = librosa.load(os.path.join(case_dir, "target.wav"), sr=None, mono=False)
        
        # Standard STFT transformation
        mix_stft = np.stack([
            librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length),
            librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)
        ])
        target_stft = np.stack([
            librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length),
            librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)
        ])
        
        # Oracle estimation
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        for algo_name, algo_instance in algos.items():
            # 1. Process
            enhanced_stft = algo_instance.process(mix_stft, oracle_rtf)
            
            # 2. Inverse STFT
            enhanced_time = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            # 3. Evaluate
            results = evaluator.evaluate_performance(target, mix, enhanced_time)
            
            # 4. Print Row
            in_db = f"{results['sisdr_in']:>7.2f}"
            out_db = f"{results['sisdr_out']:>8.2f}"
            imp_db = f"{results['improvement']:+9.2f} dB"
            print(f"{name:<25} | {algo_name:<15} | {in_db} | {out_db} | {imp_db}")

def main():
    print("--- Phase 1: Preparing Validation Set ---")
    generate_testcases()
    
    print("\n--- Phase 2: Running Algorithmic Gauntlet ---")
    run_evaluations()

if __name__ == "__main__":
    main()
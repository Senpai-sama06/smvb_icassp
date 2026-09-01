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
from algos.ulb import UniversalLinearBeamformer 

# A lightweight custom CDR to easily inject different threshold rules
class CustomThreshold_CDR:
    def __init__(self, num_mics=2, threshold=0.15, kappa=25.0, zeta_min=1e-5, zeta_max=0.5):
        self.M = num_mics
        self.epsilon_0 = threshold
        self.kappa = kappa
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ulb_engine = UniversalLinearBeamformer(num_mics=self.M)

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        _, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        d_exp = target_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        
        rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        trace_R = np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12
        
        # Continuous Sigmoid Clutch using the injected threshold
        z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.kappa * (epsilon - self.epsilon_0)))
        zeta_tensor = z_factor * trace_R
        
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        return self.ulb_engine.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)

    def apply_weights(self, stft_mix, weights):
        return self.ulb_engine.apply_weights(stft_mix, weights)

def calculate_sisdr_only(target, est, mix):
    min_len = min(len(target), len(est), len(mix))
    t, e, m = target[:min_len], est[:min_len], mix[:min_len]

    def si_sdr(ref, sig):
        ref = ref - np.mean(ref); sig = sig - np.mean(sig)
        alpha = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-12)
        target_scaled = alpha * ref
        noise_sig = sig - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise_sig, noise_sig) + 1e-12))
    
    return si_sdr(t, e) - si_sdr(t, m)

def main():
    print("--- RUNNING MISSING EXP 2: THRESHOLD SCALING PROOF ---")
    fs = 16000
    n_fft = 1024
    hop_length = 256
    
    array_sizes = [2, 4, 8]
    test_angle = 45 # A moderately challenging encroachment angle
    
    # Define the threshold rules to test
    rules = {
        'Proposed 1/M': lambda m: 1.0 / m,
        'Aperture 1/sqrt(M)': lambda m: 1.0 / np.sqrt(m),
        'Fixed Strict (0.15)': lambda m: 0.15,
        'Fixed Loose (0.50)': lambda m: 0.50
    }
    
    results = {rule: [] for rule in rules.keys()}
    results['Nominal MVDR'] = []
    
    for M in array_sizes:
        print(f"\n>>> Simulating {M}-Microphone Array at {test_angle}° <<<")
        N = M - 1 
        
        simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=20, fs=fs) 
        evaluator = Evaluator(ref_mic=0)
        random.seed(42); np.random.seed(42)
        
        i_angles = [test_angle]
        if M == 4: i_angles = [test_angle, 150, 240]
        elif M == 8: i_angles = [test_angle, 120, 150, 180, 210, 240, 300]
            
        mix, target, _, _ = simulator.simulate(
            n=N, reverb=True, target_rt60=0.2, interferer_angles=i_angles, save_outputs=False
        )
        mix = mix.T; target = target.T
        
        mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)
        
        # Nominal Baseline
        ulb = UniversalLinearBeamformer(num_mics=M)
        w_nom = ulb.process(R_matrix, oracle_rtf, np.zeros((mix_stft.shape[1], mix_stft.shape[2])), zeta=1e-5)
        est_nom = librosa.istft(ulb.apply_weights(mix_stft, w_nom), hop_length=hop_length)
        results['Nominal MVDR'].append(calculate_sisdr_only(target[0], est_nom, mix[0]))
        
        # Test each threshold rule
        for rule_name, rule_func in rules.items():
            threshold_val = rule_func(M)
            cdr = CustomThreshold_CDR(num_mics=M, threshold=threshold_val)
            w_cdr = cdr.process(R_matrix, oracle_rtf)
            est_cdr = librosa.istft(cdr.apply_weights(mix_stft, w_cdr), hop_length=hop_length)
            results[rule_name].append(calculate_sisdr_only(target[0], est_cdr, mix[0]))

    # --- Plotting the Bar Chart ---
    x = np.arange(len(array_sizes))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['black', 'crimson', 'steelblue', 'orange', 'purple']
    labels = ['Nominal MVDR', 'Proposed 1/M', 'Aperture 1/sqrt(M)', 'Fixed Strict (0.15)', 'Fixed Loose (0.50)']
    
    for i, label in enumerate(labels):
        offset = (i - 2) * width
        ax.bar(x + offset, results[label], width, label=label, color=colors[i], alpha=0.85, edgecolor='black')

    ax.set_ylabel('SI-SDR Improvement (dB)', fontsize=12)
    ax.set_title('Justification of the 1/M Spatial Boundary Scaling', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}\n(Leaky $\\rightarrow$ Directive)' for m in array_sizes], fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp2_threshold_scaling_proof.png', dpi=300)
    print("\n✅ Saved 'results/exp2_threshold_scaling_proof.png'")

if __name__ == "__main__":
    main()
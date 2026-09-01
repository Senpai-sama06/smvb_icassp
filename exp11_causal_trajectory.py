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

class Trajectory_CDR:
    def __init__(self, num_mics=4, policy='continuous'):
        self.M = num_mics
        self.policy = policy 
        self.epsilon_0 = 1.0 / self.M
        self.slope = 25.0
        self.zeta_min = 1e-5
        self.zeta_max = 0.5
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)

    def process_frame(self, Ry_frame, target_rtf_frame):
        # Eigendecomposition of the single frame SCM
        evals, evecs = np.linalg.eigh(Ry_frame)
        u1 = evecs[..., -1]
        
        # Calculate Epsilon
        num = np.abs(np.vdot(target_rtf_frame, u1))
        den = np.linalg.norm(target_rtf_frame) * np.linalg.norm(u1)
        rho = np.clip(num / (den + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        
        # Mean Trace
        trace_power = (np.trace(Ry_frame).real + 1e-12) / self.M
        
        if self.policy == 'hard':
            z_factor = self.zeta_max if epsilon > self.epsilon_0 else self.zeta_min
        else: # continuous
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.slope * (epsilon - self.epsilon_0)))
            
        zeta_val = z_factor * trace_power
        
        # We need a 3D matrix (1, 1, M, M) to feed the ULB engine for a single frame
        Ry_3d = Ry_frame[np.newaxis, np.newaxis, :, :]
        rtf_3d = target_rtf_frame[np.newaxis, :]
        mu_inv = np.zeros((1, 1), dtype=np.float32)
        zeta_3d = np.array([[zeta_val]])
        
        w = self.ulb.process(Ry_3d, rtf_3d, mu_inv, zeta=zeta_3d)
        return w[0, 0, :], epsilon, z_factor

def main():
    print("--- RUNNING EXP 11: CAUSAL TRAJECTORY ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4
    
    # We will simulate a moving interferer by generating discrete STFT chunks at different angles
    angles = np.linspace(30, 88, 60) # Sweeping towards the 90 degree target
    frames_per_angle = 5 
    
    eps_history = []
    zeta_hard = []
    zeta_cont = []
    w_hard = []
    w_cont = []

    print(">>> Simulating Moving Interferer (30 to 88 degrees) <<<")
    
    # Pre-generate target speech
    sim_base = AcousticSceneSimulator(num_mics=M, snr_target_db=20, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    prev_Ry = None
    
    for theta in angles:
        # Simulate a short chunk of audio at this specific angle
        mix, target, _, _ = sim_base.simulate(n=1, reverb=False, interferer_angles=[theta], save_outputs=False)
        mix = mix.T[:, :hop_length * frames_per_angle] 
        target = target.T[:, :hop_length * frames_per_angle]
        
        mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
        
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = np.matmul(y, y_conj) # Instantaneous SCM
        
        cdr_hard = Trajectory_CDR(num_mics=M, policy='hard')
        cdr_cont = Trajectory_CDR(num_mics=M, policy='continuous')
        
        # R_matrix shape is (F, T, M, M), so loop over T (axis 1)
        for t in range(R_matrix.shape[1]):
            Ry_f = R_matrix[10, t, :, :] # Frequency bin 10
            rtf_f = oracle_rtf[:, 10]    # Shape is (M, F) -> Extract (M,)
            
            # Add some temporal smoothing to the SCM to simulate real tracking
            if prev_Ry is not None:
                Ry_f = 0.9 * prev_Ry + 0.1 * Ry_f
            prev_Ry = Ry_f
            
            w_h, eps, z_h = cdr_hard.process_frame(Ry_f, rtf_f)
            w_c, _, z_c = cdr_cont.process_frame(Ry_f, rtf_f)
            
            eps_history.append(eps)
            zeta_hard.append(z_h)
            zeta_cont.append(z_c)
            w_hard.append(w_h)
            w_cont.append(w_c)

    # Compute frame-to-frame Jitter
    w_hard = np.array(w_hard)
    w_cont = np.array(w_cont)
    
    jitter_hard = np.sum(np.abs(w_hard[1:, :] - w_hard[:-1, :])**2, axis=-1)
    jitter_cont = np.sum(np.abs(w_cont[1:, :] - w_cont[:-1, :])**2, axis=-1)
    
    time_frames = np.arange(len(eps_history))
    
    # Dynamically interpolate the angle array to exactly match the number of STFT frames
    angle_history = np.linspace(angles[0], angles[-1], len(time_frames))
    
    # Pad jitter by 1 frame to match time_frames length
    jitter_hard = np.pad(jitter_hard, (1, 0), mode='edge')
    jitter_cont = np.pad(jitter_cont, (1, 0), mode='edge')

    # --- Plotting the 4-Panel Mechanism ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # 1. Encroachment Angle
    axs[0].plot(time_frames, angle_history, 'k-', linewidth=3)
    axs[0].set_ylabel(r'Interferer Angle $\theta_i$ (°)', fontsize=12, fontweight='bold')
    axs[0].set_title('(a) Spatial State (Moving Interferer encroaching on 90° Target)', fontsize=12)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    
    # 2. Control Variable (Epsilon)
    axs[1].plot(time_frames, eps_history, 'steelblue', linewidth=2)
    axs[1].axhline(1.0/M, color='gray', linestyle='--', label=r'Boundary $\epsilon_0 = 1/M$')
    axs[1].set_ylabel(r'Leakage Metric $\epsilon(t)$', fontsize=12, fontweight='bold')
    axs[1].set_title('(b) Control Variable (Spatial Subspace Compatibility)', fontsize=12)
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle=':', alpha=0.7)
    
    # 3. Adaptation Policy (Zeta)
    axs[2].plot(time_frames, zeta_hard, 'lightcoral', linewidth=2, label='Hard Threshold', alpha=0.8)
    axs[2].plot(time_frames, zeta_cont, 'crimson', linewidth=3, label='Continuous Sigmoid (CDR)')
    axs[2].set_ylabel(r'Regularization $\zeta(t)$', fontsize=12, fontweight='bold')
    axs[2].set_title('(c) Adaptation Policy', fontsize=12)
    axs[2].legend(loc='upper left')
    axs[2].grid(True, linestyle=':', alpha=0.7)
    
    # 4. Filter Jitter
    axs[3].semilogy(time_frames, jitter_hard, 'lightcoral', linewidth=2, label='Hard Threshold Jitter', alpha=0.8)
    axs[3].semilogy(time_frames, jitter_cont, 'crimson', linewidth=2, label='Continuous Jitter (CDR)')
    axs[3].set_ylabel(r'$\|\Delta \mathbf{w}(t)\|_2^2$', fontsize=12, fontweight='bold')
    axs[3].set_title('(d) Resulting Spatial Filter Variation', fontsize=12)
    axs[3].set_xlabel('Time (Frames)', fontsize=12, fontweight='bold')
    axs[3].legend(loc='upper left')
    axs[3].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp11_causal_trajectory.png', dpi=300)
    print("\n✅ Saved 'results/exp11_causal_trajectory.png'")

if __name__ == "__main__":
    main()
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pathlib import Path

# --- IEEE Formatting Tweaks ---
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14
})

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer

class Policy_Ablation_CDR:
    def __init__(self, num_mics=4, policy='continuous'):
        self.M = num_mics
        self.policy = policy
        self.zeta_min = 1e-5
        self.zeta_max = 0.5
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)
        self.slope = 25.0
        self.epsilon_0 = 1.0 / self.M

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        evals, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        # Proposed Epsilon Control Variable
        d_exp = target_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        
        trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / self.M

        # Adaptation Policy
        if self.policy == 'hard':
            z_factor = np.where(epsilon > self.epsilon_0, self.zeta_max, self.zeta_min)
        elif self.policy == 'continuous':
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.slope * (epsilon - self.epsilon_0)))

        zeta_tensor = z_factor * trace_power
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        weights = self.ulb.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        return weights

    def apply_weights(self, stft_mix, weights):
        return self.ulb.apply_weights(stft_mix, weights)

def calculate_sisdr(target, est, mix):
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
    print("--- GENERATING FIG 4: CONTINUOUS VS HARD ADAPTATION ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4 
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    # Encroachment scenario to trigger the transition policy
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False)
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    policies = ['hard', 'continuous']
    labels = [r'$\epsilon$ + Hard Threshold', r'$\epsilon$ + Continuous (CDR)']
    
    res_sisdr = []
    res_jitter = []
    
    for pol in policies:
        cdr = Policy_Ablation_CDR(num_mics=M, policy=pol)
        weights = cdr.process(R_matrix, oracle_rtf)
        est = librosa.istft(cdr.apply_weights(mix_stft, weights), hop_length=hop_length)
        
        res_sisdr.append(calculate_sisdr(target[0], est, mix[0]))
        res_jitter.append(np.mean(np.sum(np.abs(weights[:, 1:, :] - weights[:, :-1, :])**2, axis=-1)))

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    colors = ['#cccccc', '#4682b4']
    width = 0.5
    
    # Panel (a): SI-SDRi
    axs[0].bar(labels, res_sisdr, color=colors, edgecolor='black', width=width, zorder=3)
    axs[0].set_title('(a) Speech Quality (SI-SDRi)', fontweight='bold')
    axs[0].set_ylabel('dB')
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # Panel (b): Filter Jitter
    axs[1].bar(labels, res_jitter, color=colors, edgecolor='black', width=width, zorder=3)
    axs[1].set_yscale('log')
    axs[1].set_title(r'(b) Filter Jitter ($\|\Delta \mathbf{w}\|_2^2$)', fontweight='bold')
    axs[1].set_ylabel('Log Scale')
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    plt.suptitle('Adaptation Policy Ablation', fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/fig4_continuous_vs_hard.png', dpi=1200, bbox_inches='tight')
    print("\n✅ Saved 'results/fig4_continuous_vs_hard.pdf'")

if __name__ == "__main__":
    main()
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
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer

class Factorial_Ablation_CDR:
    def __init__(self, num_mics=4, control='epsilon', policy='continuous', zeta_min=1e-5, zeta_max=0.5):
        self.M = num_mics
        self.control = control
        self.policy = policy
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ulb = UniversalLinearBeamformer(num_mics=self.M)
        self.slope = 25.0
        self.epsilon_0 = 1.0 / self.M

    def process(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        evals, evecs = np.linalg.eigh(R_matrix)
        u1 = evecs[..., -1]
        
        trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / self.M
        z_factor = np.full((F, T), self.zeta_min, dtype=np.float32)

        if self.control == 'static':
            if self.policy == 'nominal':
                z_factor.fill(self.zeta_min)
            elif self.policy == 'sledgehammer':
                z_factor.fill(self.zeta_max)
        else:
            if self.control == 'epsilon':
                d_exp = target_rtf[:, np.newaxis, :]
                num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
                d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
                u1_norm = np.linalg.norm(u1, axis=-1)
                rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
                metric = 1.0 - (rho ** 2)
                threshold = self.epsilon_0
                
            elif self.control == 'kappa':
                lambda_max = np.abs(evals[..., -1])
                lambda_min = np.abs(evals[..., 0]) + 1e-12
                log_kappa = np.log10(lambda_max / lambda_min + 1e-12)
                k_min, k_max = np.percentile(log_kappa, 5), np.percentile(log_kappa, 95)
                metric = np.clip((log_kappa - k_min) / (k_max - k_min + 1e-12), 0.0, 1.0)
                threshold = np.median(metric)

            if self.policy == 'hard':
                z_factor = np.where(metric > threshold, self.zeta_max, self.zeta_min)
            elif self.policy == 'continuous':
                z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.slope * (metric - threshold)))

        zeta_tensor = z_factor * trace_power
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        weights = self.ulb.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        return weights, z_factor

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
    print("--- GENERATING FIG 2: FACTORIAL 2x2 ABLATION (IEEE PDF) ---")
    fs, n_fft, hop_length, M = 16000, 1024, 256, 4 
    
    simulator = AcousticSceneSimulator(num_mics=M, snr_target_db=15, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    
    mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[75], save_outputs=False)
    mix = mix.T; target = target.T
    mix_stft = np.stack([librosa.stft(mix[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    target_stft = np.stack([librosa.stft(target[m], n_fft=n_fft, hop_length=hop_length) for m in range(M)])
    oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
    
    y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
    y_conj = y.conj().transpose(0, 1, 3, 2)
    R_matrix = lfilter([1 - 0.98], [1, -0.98], np.matmul(y, y_conj), axis=1)

    configs = [
        ('static', 'nominal'), ('static', 'sledgehammer'),
        ('kappa', 'hard'), ('kappa', 'continuous'),
        ('epsilon', 'hard'), ('epsilon', 'continuous')
    ]
    
    res = {}
    for ctrl, pol in configs:
        cdr = Factorial_Ablation_CDR(num_mics=M, control=ctrl, policy=pol)
        weights, z_factor = cdr.process(R_matrix, oracle_rtf)
        est = librosa.istft(cdr.apply_weights(mix_stft, weights), hop_length=hop_length)
        
        res[f"{ctrl}_{pol}"] = {
            'sisdr': calculate_sisdr(target[0], est, mix[0]),
            'jitter': np.mean(np.sum(np.abs(weights[:, 1:, :] - weights[:, :-1, :])**2, axis=-1)),
            'p_sat': np.mean(z_factor > 0.9 * 0.5) * 100,
            'std_z': np.std(z_factor)
        }

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(8.5, 6.5)) # Scaled better for IEEE columns
    
    groups = ['kappa', 'epsilon']
    # Added newlines to prevent overlap
    labels = ['Condition Number\nBaseline ($\kappa$)', 'Spatial Subspace\nProposed ($\epsilon$)'] 
    x = np.arange(len(groups))
    width = 0.35
    
    metrics = [
        ('sisdr', '(a) SI-SDRi', 'dB', axs[0, 0], False),
        ('p_sat', r'(b) Saturation Fraction ($P_{sat} > 90\%$)', '% of frames', axs[0, 1], False),
        ('std_z', r'(c) Std. Deviation of $\zeta$', 'Standard Deviation', axs[1, 0], False),
        ('jitter', r'(d) Filter Jitter ($\|\Delta \mathbf{w}\|_2^2$)', 'Log Scale', axs[1, 1], True)
    ]
    
    for key, title, ylabel, ax, is_log in metrics:
        hard_vals = [res[f"{g}_hard"][key] for g in groups]
        cont_vals = [res[f"{g}_continuous"][key] for g in groups]
        
        bar1 = ax.bar(x - width/2, hard_vals, width, label='Hard Threshold', color='#cccccc', edgecolor='black', zorder=3)
        bar2 = ax.bar(x + width/2, cont_vals, width, label='Continuous Policy', color='#4682b4', edgecolor='black', zorder=3)
        
        # Only plot reference lines on relevant axes to avoid 0-line clutter
        if key in ['sisdr', 'jitter']:
            line1 = ax.axhline(res['static_nominal'][key], color='forestgreen', linestyle='--', linewidth=1.5, label='Nominal MVDR', zorder=4)
            line2 = ax.axhline(res['static_sledgehammer'][key], color='crimson', linestyle=':', linewidth=1.5, label='Fixed Heavy Load', zorder=4)
        elif key == 'p_sat':
            line2 = ax.axhline(res['static_sledgehammer'][key], color='crimson', linestyle=':', linewidth=1.5, label='Fixed Heavy Load', zorder=4)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
        
        if is_log: 
            ax.set_yscale('log')
            ax.set_ylim(bottom=5e-4) # Give some breathing room to the jitter plot
            
    # Add a single global legend at the top of the figure
    handles = [bar1, bar2]
    if 'line1' in locals(): handles.append(line1)
    if 'line2' in locals(): handles.append(line2)
    
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, edgecolor='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Leave space at the top for the legend
    
    os.makedirs("results", exist_ok=True)
    # Save as PDF for vector graphics integration in LaTeX
    plt.savefig('results/fig2_factorial_ablation.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved 'results/fig2_factorial_ablation.pdf'")

if __name__ == "__main__":
    main()
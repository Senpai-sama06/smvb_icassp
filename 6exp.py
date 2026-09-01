import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from scipy.signal import lfilter

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator

# --- THEORETICAL NARROWBAND FUNCTIONS ---
def get_steering_vector(M, angle_deg, d=0.04, fs=16000, f=2000, c=343.0):
    wavelength = c / f
    k = 2 * np.pi / wavelength
    angles_rad = np.radians(angle_deg)
    tau = np.arange(M) * d * np.cos(angles_rad)
    return np.exp(-1j * k * tau)

def get_wng(w):
    w_norm_sq = np.real(np.vdot(w, w))
    return -10 * np.log10(w_norm_sq + 1e-12)

# --- EMPIRICAL STFT FUNCTIONS ---
def get_empirical_R_and_d(target_wav, mix_wav, M, target_angle, n_fft=1024, fs=16000, f_target=2000):
    """Extracts empirical covariance matrix and theoretical RTF for a specific frequency bin."""
    f_bin = int(f_target * n_fft / fs) # Bin 128 for 2kHz
    
    mix_stft = np.stack([librosa.stft(mix_wav[m], n_fft=n_fft, hop_length=256) for m in range(M)])
    target_stft = np.stack([librosa.stft(target_wav[m], n_fft=n_fft, hop_length=256) for m in range(M)])
    
    # Isolate the specific frequency bin across all time frames: shape (M, T)
    y_bin = mix_stft[:, f_bin, :] 
    t_bin = target_stft[:, f_bin, :]
    
    # Filter for active speech frames (energy > median) to avoid silence corrupting the matrix
    energy = np.sum(np.abs(t_bin)**2, axis=0)
    active_frames = energy > np.median(energy)
    y_active = y_bin[:, active_frames]
    
    # Empirical Spatial Covariance Matrix
    R_y_emp = (y_active @ y_active.conj().T) / (y_active.shape[1] + 1e-12)
    
    # Normalize trace to match theoretical scaling
    R_y_emp = R_y_emp / (np.trace(R_y_emp) + 1e-12) * M 
    
    d_theory = get_steering_vector(M, target_angle, f=f_target)
    return R_y_emp, d_theory

def main():
    print("--- RUNNING HYBRID THEORETICAL VS EMPIRICAL DIAGNOSTICS ---")
    M = 4
    fs = 16000
    sigma_n2 = 1e-3 

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # ==========================================
    # EXPERIMENT 1: Geometry vs. Matrix Breakdown
    # ==========================================
    print("Simulating Exp 1 (Geometry)...")
    target_angle = 90
    interferer_angles_th = np.linspace(90.1, 180, 100)
    
    # 1. Theoretical Curves
    delta_thetas, rhos, kappas, wngs = [], [], [], []
    d_target = get_steering_vector(M, target_angle)
    for i_angle in interferer_angles_th:
        delta_thetas.append(np.abs(target_angle - i_angle))
        d_interf = get_steering_vector(M, i_angle)
        R_y = np.outer(d_target, d_target.conj()) + np.outer(d_interf, d_interf.conj()) + sigma_n2 * np.eye(M)
        evals, evecs = np.linalg.eigh(R_y)
        rhos.append(np.abs(np.vdot(d_target, evecs[:, -1])) / (np.linalg.norm(d_target) * np.linalg.norm(evecs[:, -1])))
        kappas.append(10 * np.log10(np.abs(evals[-1] / (evals[0] + 1e-12))))
        R_inv = np.linalg.inv(R_y)
        wngs.append(get_wng(R_inv @ d_target / (d_target.conj().T @ R_inv @ d_target)))

    # 2. Empirical Scatter Points
    emp_angles = [92, 100, 120, 150, 180]
    emp_dts, emp_rhos, emp_kappas, emp_wngs = [], [], [], []
    
    sim = AcousticSceneSimulator(num_mics=M, fs=fs)
    for i_angle in emp_angles:
        np.random.seed(42)
        mix, target, _, _ = sim.simulate(n=1, reverb=False, interferer_angles=[i_angle], save_outputs=False)
        R_y_emp, d_t = get_empirical_R_and_d(target.T, mix.T, M, target_angle)
        
        # Add ambient loading to empirical matrix to ensure invertibility matches theory
        R_y_emp += (sigma_n2 * np.trace(R_y_emp)) * np.eye(M) 
        
        evals, evecs = np.linalg.eigh(R_y_emp)
        emp_rhos.append(np.abs(np.vdot(d_t, evecs[:, -1])) / (np.linalg.norm(d_t) * np.linalg.norm(evecs[:, -1])))
        emp_kappas.append(10 * np.log10(np.abs(evals[-1] / (evals[0] + 1e-12))))
        R_inv = np.linalg.inv(R_y_emp)
        emp_wngs.append(get_wng(R_inv @ d_t / (d_t.conj().T @ R_inv @ d_t)))
        emp_dts.append(np.abs(target_angle - i_angle))

    ax = axs[0]
    ax.plot(delta_thetas, rhos, 'b-', alpha=0.6, label='Theory $\\rho$')
    ax.scatter(emp_dts, emp_rhos, color='blue', marker='*', s=150, label='Empirical $\\rho$')
    ax.set_ylabel('Collinearity $\\rho$', color='b')
    ax.set_xlabel('Angular Separation $\\Delta\\theta$ (°)')
    
    ax2 = ax.twinx()
    ax2.plot(delta_thetas, kappas, 'r--', alpha=0.4, label='Theory $\\kappa$')
    ax2.scatter(emp_dts, emp_kappas, color='red', marker='s', s=80, label='Empirical $\\kappa$')
    ax2.plot(delta_thetas, wngs, 'g:', alpha=0.6, label='Theory WNG')
    ax2.scatter(emp_dts, emp_wngs, color='green', marker='o', s=80, label='Empirical WNG')
    ax2.set_ylabel('dB', color='k')
    ax2.invert_xaxis()
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=8)
    ax.set_title('Exp 1: Theory vs Empirical Broadband')

    # ==========================================
    # EXPERIMENT 2: Regularization Trade-off
    # ==========================================
    print("Simulating Exp 2 (Trade-off)...")
    d_i = get_steering_vector(M, 85)
    R_y_fixed = np.outer(d_target, d_target.conj()) + np.outer(d_i, d_i.conj()) + sigma_n2 * np.eye(M)
    zetas_th = np.logspace(-6, 1, 50)
    wng_th, null_th = [], []
    
    for z in zetas_th:
        R_loaded = R_y_fixed + z * np.trace(R_y_fixed) * np.eye(M)
        R_inv = np.linalg.inv(R_loaded)
        w = R_inv @ d_target / (d_target.conj().T @ R_inv @ d_target)
        wng_th.append(get_wng(w))
        null_th.append(20 * np.log10(np.abs(np.vdot(w, d_i)) + 1e-12))
        
    np.random.seed(42)
    mix, target, interferer, _ = sim.simulate(n=1, reverb=False, interferer_angles=[85], save_outputs=False)
    R_y_emp, d_t = get_empirical_R_and_d(target.T, mix.T, M, 90)
    
    emp_zetas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    emp_wng_trade, emp_null_trade = [], []
    for z in emp_zetas:
        R_loaded = R_y_emp + z * np.trace(R_y_emp) * np.eye(M)
        R_inv = np.linalg.inv(R_loaded)
        w = R_inv @ d_t / (d_t.conj().T @ R_inv @ d_t)
        emp_wng_trade.append(get_wng(w))
        emp_null_trade.append(20 * np.log10(np.abs(np.vdot(w, get_steering_vector(M, 85))) + 1e-12))
        
    ax = axs[1]
    ax.semilogx(zetas_th, wng_th, 'g-', alpha=0.6, label='Theory WNG')
    ax.scatter(emp_zetas, emp_wng_trade, color='green', marker='*', s=150, label='Empirical WNG')
    ax.set_xlabel('Trace-Relative Regularization ($\\zeta$)')
    ax.set_ylabel('WNG (dB)', color='g')
    
    ax3 = ax.twinx()
    ax3.semilogx(zetas_th, null_th, 'r-', alpha=0.6, label='Theory Null')
    ax3.scatter(emp_zetas, emp_null_trade, color='red', marker='s', s=80, label='Empirical Null')
    ax3.set_ylabel('Interferer Gain (dB)', color='r')
    ax.set_title('Exp 2: Empirical Trade-off Overlay')

    # ==========================================
    # EXPERIMENT 4: Robustness to SIR
    # ==========================================
    print("Simulating Exp 4 (SIR)...")
    d_i_safe = get_steering_vector(M, 45)
    sirs_th = np.linspace(-20, 20, 50)
    rhos_th = []
    for sir in sirs_th:
        p_i = 10 ** (-sir / 10)
        R_sir = np.outer(d_target, d_target.conj()) + p_i * np.outer(d_i_safe, d_i_safe.conj()) + sigma_n2 * np.eye(M)
        evals, evecs = np.linalg.eigh(R_sir)
        rhos_th.append(np.abs(np.vdot(d_target, evecs[:, -1])) / (np.linalg.norm(d_target) * np.linalg.norm(evecs[:, -1])))

    emp_sirs = [-20, -10, 0, 10, 20]
    emp_rhos_sir = []
    
    np.random.seed(42)
    # Generate separated sources to manually scale SIR
    _, target_wav, interf_wav, _ = sim.simulate(n=1, reverb=False, interferer_angles=[45], save_outputs=False)
    target_wav = target_wav.T; interf_wav = interf_wav.T
    
    # Measure baseline energy to scale appropriately
    e_t = np.sum(target_wav**2)
    e_i = np.sum(interf_wav**2)
    
    for sir in emp_sirs:
        # Scale interferer waveform to achieve desired target SIR
        desired_e_i = e_t / (10 ** (sir / 10))
        scale_factor = np.sqrt(desired_e_i / e_i)
        scaled_interf = interf_wav * scale_factor
        
        mix_scaled = target_wav + scaled_interf
        R_y_emp, d_t = get_empirical_R_and_d(target_wav, mix_scaled, M, 90)
        
        evals, evecs = np.linalg.eigh(R_y_emp)
        emp_rhos_sir.append(np.abs(np.vdot(d_t, evecs[:, -1])) / (np.linalg.norm(d_t) * np.linalg.norm(evecs[:, -1])))

    ax = axs[2]
    ax.plot(sirs_th, rhos_th, 'b-', alpha=0.6, label='Theory $\\rho$')
    ax.scatter(emp_sirs, emp_rhos_sir, color='blue', marker='*', s=150, label='Empirical STFT $\\rho$')
    ax.set_title('Exp 4: Simulated Speech SIR vs $\\rho$')
    ax.set_xlabel('Signal-to-Interference Ratio (dB)')
    ax.set_ylabel('Collinearity ($\\rho$)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('exp_theory_vs_empirical.png', dpi=300)
    print("✅ Saved 'exp_theory_vs_empirical.png'")

if __name__ == "__main__":
    main()
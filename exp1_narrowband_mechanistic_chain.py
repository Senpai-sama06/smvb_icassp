import numpy as np
import matplotlib.pyplot as plt
import os

def get_steering_vector(M, angle_deg, d=0.04, f=2000, c=343.0):
    k = 2 * np.pi / (c / f)
    tau = np.arange(M) * d * np.cos(np.radians(angle_deg))
    return np.exp(-1j * k * tau)

def main():
    print("--- RUNNING CORRECTED EXP 1: NARROWBAND MISMATCH MECHANISM ---")
    
    M = 4
    f = 2000
    true_target_angle = 90
    d_true = get_steering_vector(M, true_target_angle, f=f)
    
    # Sweep Steering Mismatch Error (0 to 15 degrees)
    mismatch_errors = np.linspace(0, 15, 500)
    
    noise_proj_vals = []
    wng_vals = []
    eps_vals = []
    zeta_vals = []
    
    sigma_n2 = 1e-4   
    sigma_T2 = 1.0    
    
    # Static Covariance Matrix (True target is fixed at 90°)
    R_y = sigma_T2 * np.outer(d_true, d_true.conj()) + sigma_n2 * np.eye(M)
    evals, evecs = np.linalg.eigh(R_y)
    u1 = evecs[:, -1]
    
    # Projection Matrix into Noise Subspace
    P_n = np.eye(M) - np.outer(u1, u1.conj()) / (np.linalg.norm(u1)**2)
    
    epsilon_0 = 1.0 / M
    kappa = 25.0
    zeta_min = 1e-5
    zeta_max = 0.5
    
    for err in mismatch_errors:
        assumed_angle = true_target_angle + err
        d_assumed = get_steering_vector(M, assumed_angle, f=f)
        
        # 1. Target Projection into Noise Subspace (The physical cause of failure)
        noise_proj = np.real(np.vdot(d_assumed, P_n @ d_assumed) / np.vdot(d_assumed, d_assumed))
        noise_proj_vals.append(noise_proj)
        
        # 2. Compute Theoretical WNG Collapse (Nominal MVDR using Assumed Vector)
        R_inv = np.linalg.inv(R_y)
        w_mvdr = R_inv @ d_assumed / (d_assumed.conj().T @ R_inv @ d_assumed + 1e-12)
        wng = -10 * np.log10(np.real(np.vdot(w_mvdr, w_mvdr)) + 1e-12)
        wng_vals.append(wng)
        
        # 3. Compute Orthogonal Subspace Leakage Metric
        num = np.abs(np.vdot(d_assumed, u1))
        den = np.linalg.norm(d_assumed) * np.linalg.norm(u1)
        rho = np.clip(num / (den + 1e-12), 0.0, 1.0)
        eps = 1.0 - rho**2
        eps_vals.append(eps)
        
        # 4. Apply Continuous Sigmoid Clutch
        zeta = zeta_min + (zeta_max - zeta_min) / (1.0 + np.exp(-kappa * (eps - epsilon_0)))
        zeta_vals.append(zeta)

    # --- Plotting the 4-Panel Mechanism ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle('Figure 1: Narrowband Analysis of the Steering Mismatch Failure Mechanism', fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 1: Noise Subspace Projection
    axs[0, 0].plot(mismatch_errors, noise_proj_vals, 'k-', linewidth=3.0)
    axs[0, 0].set_title('(a) Target Projection into Noise Subspace', fontweight='bold', fontsize=14)
    axs[0, 0].set_xlabel('Steering Mismatch Error $\Delta\\theta$ (°)', fontsize=12)
    axs[0, 0].set_ylabel('Subspace Projection Error', fontsize=12)
    axs[0, 0].grid(True, linestyle=':', alpha=0.7)
    
    # Panel 2: WNG Collapse
    axs[0, 1].plot(mismatch_errors, wng_vals, 'sienna', linewidth=3.0)
    axs[0, 1].set_title('(b) White Noise Gain Collapse', fontweight='bold', fontsize=14)
    axs[0, 1].set_xlabel('Steering Mismatch Error $\Delta\\theta$ (°)', fontsize=12)
    axs[0, 1].set_ylabel('Nominal MVDR WNG (dB)', fontsize=12)
    axs[0, 1].grid(True, linestyle=':', alpha=0.7)
    
    # Panel 3: Leakage Proxy
    axs[1, 0].plot(mismatch_errors, eps_vals, 'steelblue', linewidth=3.0)
    axs[1, 0].axhline(epsilon_0, color='gray', linestyle='--', linewidth=2, label=f'Boundary $\epsilon_0=1/M$')
    axs[1, 0].set_title('(c) Derived Orthogonal Subspace Leakage ($\epsilon$)', fontweight='bold', fontsize=14)
    axs[1, 0].set_xlabel('Steering Mismatch Error $\Delta\\theta$ (°)', fontsize=12)
    axs[1, 0].set_ylabel('Leakage $\epsilon$', fontsize=12)
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].grid(True, linestyle=':', alpha=0.7)
    
    # Panel 4: Mapping Function
    axs[1, 1].semilogy(eps_vals, zeta_vals, 'crimson', linewidth=3.0)
    axs[1, 1].axvline(epsilon_0, color='gray', linestyle='--', linewidth=2, label=f'Boundary $\epsilon_0=1/M$')
    axs[1, 1].set_title('(d) Continuous Adaptive Regularization ($\zeta$)', fontweight='bold', fontsize=14)
    axs[1, 1].set_xlabel('Subspace Leakage $\epsilon$', fontsize=12)
    axs[1, 1].set_ylabel('Trace-Relative Regularization $\zeta$', fontsize=12)
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/exp1_narrowband_mismatch_chain.png', dpi=300)
    print("✅ Saved 'results/exp1_narrowband_mismatch_chain.png'")

if __name__ == '__main__':
    main()
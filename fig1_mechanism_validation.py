import numpy as np
import matplotlib.pyplot as plt
import os

def steering_vector(angle_deg, M=4, d=0.08, f=2000, c=343.0):
    """Generates a narrowband ULA steering vector."""
    angle_rad = np.radians(angle_deg)
    # Using standard ULA phase delay formulation
    delays = np.arange(M) * (d / c) * np.cos(angle_rad)
    return np.exp(-1j * 2 * np.pi * f * delays)

def main():
    print("--- GENERATING FIG 1: MECHANISM VALIDATION ---")
    M = 4
    snr_db = 15
    sir_db = 0
    
    sigma_n2 = 1.0
    sigma_s2 = sigma_n2 * (10 ** (snr_db / 10))
    sigma_i2 = sigma_s2 * (10 ** (-sir_db / 10))
    
    target_angle = 90.0
    d_target = steering_vector(target_angle, M=M)
    
    # Sweep angular separation from 90 degrees down to 0 degrees
    delta_thetas = np.linspace(90, 0, 100)
    
    epsilons = []
    kappas = []
    wngs = []
    
    for delta in delta_thetas:
        interferer_angle = target_angle - delta
        a_i = steering_vector(interferer_angle, M=M)
        
        # Exact Analytical Spatial Covariance Matrix
        R_y = sigma_s2 * np.outer(d_target, d_target.conj()) + \
              sigma_i2 * np.outer(a_i, a_i.conj()) + \
              sigma_n2 * np.eye(M)
              
        # 1. Condition Number
        evals, evecs = np.linalg.eigh(R_y)
        kappa = np.abs(evals[-1]) / (np.abs(evals[0]) + 1e-12)
        kappas.append(kappa)
        
        # 2. Epsilon (Spatial Subspace Leakage)
        u1 = evecs[:, -1]
        num = np.abs(np.vdot(d_target, u1))
        den = np.linalg.norm(d_target) * np.linalg.norm(u1)
        rho = np.clip(num / (den + 1e-12), 0.0, 1.0)
        epsilons.append(1.0 - rho**2)
        
        # 3. White Noise Gain (WNG) of Nominal MVDR
        R_inv = np.linalg.inv(R_y)
        w_mvdr = R_inv @ d_target / (d_target.conj().T @ R_inv @ d_target)
        # WNG = 10 * log10( 1 / ||w||^2 )
        wng_db = -10 * np.log10(np.linalg.norm(w_mvdr)**2)
        wngs.append(wng_db)

    # --- Plotting the 3-Panel Figure ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # (a) Epsilon
    axs[0].plot(delta_thetas, epsilons, 'steelblue', linewidth=3)
    axs[0].set_title(r'(a) Spatial-Subspace Projection Error ($\epsilon$)', fontsize=14, fontweight='bold')
    axs[0].set_ylabel(r'$\epsilon$', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].invert_xaxis() # Read left-to-right as geometry degrades (separation -> 0)
    
    # (b) Condition Number (Log Scale)
    axs[1].semilogy(delta_thetas, kappas, 'darkorange', linewidth=3)
    axs[1].set_title(r'(b) Matrix Condition Number ($\kappa(\mathbf{R}_y)$)', fontsize=14, fontweight='bold')
    axs[1].set_ylabel(r'$\kappa$ (Log Scale)', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # (c) White Noise Gain
    axs[2].plot(delta_thetas, wngs, 'crimson', linewidth=3)
    axs[2].set_title(r'(c) Array White Noise Gain (WNG)', fontsize=14, fontweight='bold')
    axs[2].set_ylabel('WNG (dB)', fontsize=12)
    axs[2].set_xlabel(r'Angular Separation $\Delta\theta$ (°)', fontsize=14, fontweight='bold')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/fig1_mechanism_validation.png', dpi=300)
    print("\n✅ Saved 'results/fig1_mechanism_validation.png'")

if __name__ == "__main__":
    main()
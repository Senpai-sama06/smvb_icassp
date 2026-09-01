import numpy as np
import matplotlib.pyplot as plt

def get_steering_vector(M, angle_deg, d=0.04, f=2000, c=343.0):
    k = 2 * np.pi / (c / f)
    tau = np.arange(M) * d * np.cos(np.radians(angle_deg))
    return np.exp(-1j * k * tau)

def get_wng(w):
    return -10 * np.log10(np.real(np.vdot(w, w)) + 1e-12)

def main():
    print("--- RUNNING EXPERIMENT 6: TEMPORAL TRACKING ---")
    M = 4
    frames = 400
    target_angle = 90
    
    # 1. Simulate Dynamic Sweeping Interferer (30 to 150 degrees)
    interferer_angles = np.linspace(30, 150, frames)
    
    # Simulate causal STFT bins using random complex Gaussian noise (mimicking speech bursts)
    np.random.seed(42)
    s_target = (np.random.randn(frames) + 1j * np.random.randn(frames)) * 1.0 # Target speech
    s_interf = (np.random.randn(frames) + 1j * np.random.randn(frames)) * 0.8 # Slightly quieter interferer
    
    d_target = get_steering_vector(M, target_angle)
    
    # Baseline noise floor
    sigma_n2 = 1e-4 
    
    # Tracking arrays
    rhos, epsilons = [], []
    zeta_binary, zeta_cdr = [], []
    wng_nominal, wng_binary, wng_cdr = [], [], []
    
    R_y = np.eye(M) * sigma_n2 # Initialize Covariance Matrix
    alpha = 0.8 # Exponential Moving Average (EMA) factor for causal tracking
    
    # CDR Parameters
    zeta_min, zeta_max = 1e-5, 0.5
    
    for t in range(frames):
        # 2. Build instantaneous signal vector
        d_i = get_steering_vector(M, interferer_angles[t])
        y_t = s_target[t] * d_target + s_interf[t] * d_i + np.sqrt(sigma_n2/2) * (np.random.randn(M) + 1j*np.random.randn(M))
        
        # 3. Causal EMA Covariance Matrix Update
        R_y = alpha * R_y + (1 - alpha) * np.outer(y_t, y_t.conj())
        
        # 4. Extract Tracking Metrics
        evals, evecs = np.linalg.eigh(R_y)
        u1 = evecs[:, -1]
        
        rho = np.abs(np.vdot(d_target, u1)) / (np.linalg.norm(d_target) * np.linalg.norm(u1) + 1e-12)
        rho = np.clip(rho, 0.0, 1.0)
        epsilon = 1.0 - rho**2
        
        rhos.append(rho)
        epsilons.append(epsilon)
        
        trace_R = np.trace(R_y).real + 1e-12
        
        # 5. Compute Zetas for Ablations
        z_nom = zeta_min * trace_R
        z_bin = zeta_max * trace_R if epsilon > 0.4 else zeta_min * trace_R
        z_cdr = (zeta_min + zeta_max * (epsilon ** 3)) * trace_R
        
        zeta_binary.append(z_bin)
        zeta_cdr.append(z_cdr)
        
        # 6. Compute Filters and WNG
        def calc_wng(z):
            R_inv = np.linalg.inv(R_y + z * np.eye(M))
            w = R_inv @ d_target / (d_target.conj().T @ R_inv @ d_target + 1e-12)
            return get_wng(w)
            
        wng_nominal.append(calc_wng(z_nom))
        wng_binary.append(calc_wng(z_bin))
        wng_cdr.append(calc_wng(z_cdr))

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    time_axis = np.linspace(0, 5, frames) # 5 seconds
    
    # Plot 1: Geometry Context
    axs[0].plot(time_axis, [target_angle]*frames, 'k--', label='Target Angle (Fixed 90°)')
    axs[0].plot(time_axis, interferer_angles, 'r-', alpha=0.7, label='Interferer Trajectory')
    axs[0].axvspan(1.8, 3.2, color='red', alpha=0.1, label='Encroachment Danger Zone')
    axs[0].set_ylabel('Angle (°)')
    axs[0].set_title('Exp 6: Temporal Tracking of Moving Interferer')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Regularization Parameter (\zeta)
    axs[1].semilogy(time_axis, zeta_binary, 'purple', linestyle='--', label='Binary Hard-Switching $\\zeta(t)$')
    axs[1].semilogy(time_axis, zeta_cdr, 'crimson', linewidth=2.5, label='Proposed CDR $\\zeta(t)$')
    axs[1].set_ylabel('Regularization $\\zeta$ (Log Scale)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: Resulting Stability (WNG)
    axs[2].plot(time_axis, wng_nominal, 'black', linestyle=':', label='Nominal MVDR (Crashes)')
    axs[2].plot(time_axis, wng_binary, 'purple', linestyle='--', label='Binary Adaptive (Chatters)')
    axs[2].plot(time_axis, wng_cdr, 'crimson', linewidth=2.5, label='Proposed CDR (Smooth)')
    axs[2].set_ylabel('WNG (dB)')
    axs[2].set_xlabel('Time (Seconds)')
    axs[2].set_ylim(-15, 10)
    axs[2].legend(loc='lower right')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('exp6_temporal_behavior.png', dpi=300)
    print("✅ Saved 'exp6_temporal_behavior.png'")

if __name__ == "__main__":
    main()
import numpy as np
import scipy.linalg as LA

def oracle_gevd_process(target_stft, noise_stft, mix_stft, ref_mic=0):
    """
    Rigorous Oracle GEVD Beamformer (Matches pb_bss toolbox standards).
    Uses the principal generalized eigenvector with reference-mic projection.
    """
    M, F, T = target_stft.shape
    enhanced_stft = np.zeros((F, T), dtype=complex)
    
    for f in range(F):
        S = target_stft[:, f, :]
        N_noise = noise_stft[:, f, :]
        
        # Spatial covariance matrices
        Phi_xx = (S @ S.conj().T) / T
        Phi_nn = (N_noise @ N_noise.conj().T) / T
        
        # Regularize noise covariance to ensure positive-definiteness
        Phi_nn_reg = Phi_nn + 1e-6 * np.trace(Phi_nn) * np.eye(M)
        
        try:
            # Solve generalized eigenvalue problem
            eigenvalues, eigenvectors = LA.eigh(Phi_xx, Phi_nn_reg)
            
            # Extract principal eigenvector (corresponds to highest eigenvalue)
            max_idx = np.argmax(eigenvalues)
            w = eigenvectors[:, max_idx]
            
            # Robust Reference-Mic Projection (Scaling alignment)
            # w_opt = w * (w^H * Phi_nn * u_ref) / (w^H * Phi_nn * w)
            u_ref = np.zeros(M)
            u_ref[ref_mic] = 1.0
            
            numerator = w.conj().T @ Phi_nn_reg @ u_ref
            denominator = w.conj().T @ Phi_nn_reg @ w
            scale = numerator / (denominator + 1e-12)
            
            w_gevd = w * scale.conj()
            
        except LA.LinAlgError:
            # Fallback to standard Delay-and-Sum if pencil fails
            w_gevd = np.zeros(M, dtype=complex)
            w_gevd[ref_mic] = 1.0
            
        enhanced_stft[f, :] = w_gevd.conj().T @ mix_stft[:, f, :]
        
    return enhanced_stft
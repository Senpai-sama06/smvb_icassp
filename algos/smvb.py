import numpy as np
from scipy.signal import lfilter

class SMVB:
    def __init__(self, num_mics=2, alpha=0.9, tau_eta=5.0, tau_rho=0.966, gamma_max=1e-3):
        """
        Initializes the Switching Minimum Variance Beamformer (SMVB).
        Fully vectorized for offline/batch processing (Zero Python Loops).
        """
        self.M = num_mics
        self.alpha = alpha          
        self.tau_eta = tau_eta      
        self.tau_rho = tau_rho      
        self.gamma_max = gamma_max  
        
    def process(self, stft_mix, target_rtf):
        """
        stft_mix: Complex numpy array of shape (M, F, T)
        target_rtf: Complex numpy array of shape (M, F) representing d(f)
        """
        M, F, T = stft_mix.shape
        
        # 1. Prepare Tensors
        # y shape: (F, T, M, 1)
        y = stft_mix.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        
        # d_f shape: (F, 1, M, 1)
        d_f = target_rtf.T[:, np.newaxis, :, np.newaxis]
        d_f_conj = d_f.conj().transpose(0, 1, 3, 2)

        # 2. Vectorized Covariance Update (IIR Filter along Time Axis)
        R_inst = np.matmul(y, y_conj)  # Shape: (F, T, M, M)
        
        b = [1 - self.alpha]
        a = [1, -self.alpha]
        # lfilter applies the EMA instantly across the entire T dimension
        R_yy = lfilter(b, a, R_inst, axis=1)

        # 3. Batched Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R_yy)
        
        # eigh sorts ascending. max is index -1, min is index 0
        lambda_M = eigenvalues[..., 0]   # Shape: (F, T)
        lambda_1 = eigenvalues[..., -1]  # Shape: (F, T)
        u_1 = eigenvectors[..., -1]      # Shape: (F, T, M)

        # 4. Spatial State Evaluation
        eta = lambda_1 / (lambda_M + 1e-12)
        
        # Collinearity
        # np.einsum for batched dot products: d_f^H @ u_1
        num = np.abs(np.einsum('ftmc,ftm->ft', d_f_conj, u_1))
        den = np.linalg.norm(d_f.squeeze(), axis=-1)[:, np.newaxis] * np.linalg.norm(u_1, axis=-1)
        rho = num / (den + 1e-12)

        # 5. Switching Logic
        I_tf = ((eta > self.tau_eta) & (rho <= self.tau_rho)).astype(float)
        
        # Temporal smoothing of the state using lfilter
        beta = 0.8
        b_beta = [1 - beta]
        a_beta = [1, -beta]
        omega_tilde = lfilter(b_beta, a_beta, I_tf, axis=1)
        Omega_tf = omega_tilde > 0.5  # Boolean mask of shape (F, T)

        # 6. Beamformer Weight Calculation
        R_compute = np.copy(R_yy)
        starved_mask = ~Omega_tf
        
        if np.any(starved_mask):
            gamma = np.minimum(0.1 * lambda_M[starved_mask], self.gamma_max)
            # Apply diagonal loading only to starved TF bins
            M_eye = np.eye(M)
            R_compute[starved_mask] += gamma[:, np.newaxis, np.newaxis] * M_eye

        # Batched Matrix Inversion over (F * T) matrices
        inv_R = np.linalg.pinv(R_compute)

        # Compute weights: (inv_R @ d_f) / (d_f^H @ inv_R @ d_f)
        num_w = np.matmul(inv_R, d_f)                        # Shape: (F, T, M, 1)
        den_w = np.matmul(d_f_conj, num_w) + 1e-12           # Shape: (F, T, 1, 1)
        w_tf = num_w / den_w                                 # Shape: (F, T, M, 1)

        # 7. Apply Weights
        w_tf_conj = w_tf.conj().transpose(0, 1, 3, 2)        # Shape: (F, T, 1, M)
        enhanced_stft = np.matmul(w_tf_conj, y).squeeze()    # Shape: (F, T)
        
        return enhanced_stft
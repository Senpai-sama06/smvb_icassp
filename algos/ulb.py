import numpy as np

class UniversalLinearBeamformer:
    """
    The 'Treat' Module: Vectorized Universal Linear Beamformer (ULB).
    Computes parameterized spatial weights frame-by-frame without loops.
    """
    def __init__(self, num_mics=2, eps=1e-6):
        self.M = num_mics
        self.eps = eps  # Tiny diagonal loading for numerical stability during inversion

    def process(self, R_matrix, target_rtf, mu_inv, zeta=None):
        """
        Calculates the ULB weights for all Time-Frequency bins simultaneously.
        """
        # --- FIX 1: Ensure target_rtf is (F, M) ---
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T  # Transpose (M, F) to (F, M)
            
        # --- FIX 2: Handle dynamic zeta (diagonal loading) ---
        if zeta is None:
            zeta_exp = self.eps
        elif isinstance(zeta, np.ndarray):
            zeta_exp = zeta[:, :, np.newaxis, np.newaxis] # Expand (F, T) to (F, T, 1, 1)
        else:
            zeta_exp = float(zeta) # Handle scalar float for the sweep
            
        # 1. Expand dimensions for batched matrix multiplication
        d = target_rtf[:, np.newaxis, :, np.newaxis] 
        d_conj = d.conj().transpose(0, 1, 3, 2)      
        
        # 2. Invert the Covariance Matrix
        eye = np.eye(self.M, dtype=np.complex64)[np.newaxis, np.newaxis, :, :]
        # Apply the interference penalty (zeta) dynamically!
        R_stable = R_matrix + (zeta_exp * eye)
        R_inv = np.linalg.inv(R_stable)              # Output: (F, T, M, M)
        
        # 3. Calculate Numerator: R_inv * d
        num = np.matmul(R_inv, d)                    # Output: (F, T, M, 1)
        
        # 4. Calculate Denominator: mu^{-1} + d^H * R_inv * d
        d_H_R_inv = np.matmul(d_conj, R_inv)         
        d_H_R_inv_d = np.matmul(d_H_R_inv, d).real   
        
        mu_inv_exp = mu_inv[:, :, np.newaxis, np.newaxis]
        den = mu_inv_exp + d_H_R_inv_d
        
        # 5. Compute Final Weights
        w = num / den
        
        return w.squeeze(-1)

    def apply_weights(self, stft_mix, weights):
        """
        Applies the computed weights to the STFT mixture.
        stft_mix: (M, F, T)
        weights: (F, T, M)
        """
        # Transpose stft_mix to (F, T, M)
        y = stft_mix.transpose(1, 2, 0)
        
        # Element-wise multiply and sum across the microphone dimension
        enhanced_stft = np.sum(weights.conj() * y, axis=-1) # (F, T)
        return enhanced_stft
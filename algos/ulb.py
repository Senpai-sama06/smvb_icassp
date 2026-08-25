import numpy as np

class UniversalLinearBeamformer:
    """
    The 'Treat' Module: Vectorized Universal Linear Beamformer (ULB).
    Computes parameterized spatial weights frame-by-frame without loops.
    """
    def __init__(self, num_mics=2, eps=1e-6):
        self.M = num_mics
        self.eps = eps  # Tiny diagonal loading for numerical stability during inversion

    def process(self, R_matrix, target_rtf, mu_inv):
        """
        Calculates the ULB weights for all Time-Frequency bins simultaneously.
        """
        # --- FIX: Ensure target_rtf is (F, M) ---
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T  # Transpose (M, F) to (F, M)
            
        # 1. Expand dimensions for batched matrix multiplication
        # target_rtf -> (F, 1, M, 1)
        d = target_rtf[:, np.newaxis, :, np.newaxis] 
        # d_conj -> (F, 1, 1, M)
        d_conj = d.conj().transpose(0, 1, 3, 2)      
        
        # 2. Invert the Covariance Matrix
        # Add tiny identity matrix to prevent singular matrix crashes
        eye = np.eye(self.M, dtype=np.complex64)[np.newaxis, np.newaxis, :, :]
        R_stable = R_matrix + self.eps * eye
        R_inv = np.linalg.inv(R_stable)              # Output: (F, T, M, M)
        
        # 3. Calculate Numerator: R_inv * d
        num = np.matmul(R_inv, d)                    # Output: (F, T, M, 1)
        
        # 4. Calculate Denominator: mu^{-1} + d^H * R_inv * d
        # Matrix multiply d_conj with R_inv -> (F, T, 1, M)
        d_H_R_inv = np.matmul(d_conj, R_inv)         
        # Multiply result by d -> (F, T, 1, 1)
        d_H_R_inv_d = np.matmul(d_H_R_inv, d).real   # The result of d^H * R^-1 * d is strictly real
        
        # Expand mu_inv to match the tensor shape -> (F, T, 1, 1)
        mu_inv_exp = mu_inv[:, :, np.newaxis, np.newaxis]
        
        den = mu_inv_exp + d_H_R_inv_d
        
        # 5. Compute Final Weights
        w = num / den
        
        # Remove the dummy last dimension -> (F, T, M)
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
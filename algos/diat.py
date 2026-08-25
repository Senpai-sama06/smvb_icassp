import numpy as np
from algos.ulb import UniversalLinearBeamformer

class DIAT_Bf:
    """
    Dynamic Identify and Treat Beamformer (DIAT-Bf).
    Autonomously coordinates spatial physics into WNG regularization.
    """
    def __init__(self, num_mics=2, rho_0=0.85, kappa=20.0, zeta_min=1e-6, zeta_max=1e-2):
        self.M = num_mics
        self.rho_0 = rho_0
        self.kappa = kappa
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        
        # Initialize the mathematically optimal Treat engine
        self.ulb_engine = UniversalLinearBeamformer(num_mics=self.M)

    def identify(self, R_matrix, target_rtf):
        """
        The 'Identify' Block: 
        Calculates instantaneous rho(t,f) and maps it to zeta(t,f).
        """
        F, T, M, _ = R_matrix.shape
        
        # Ensure target_rtf is shape (F, M)
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        # 1. Eigendecomposition of the instantaneous covariance matrix
        # Returns eigenvalues and eigenvectors for all F and T simultaneously
        _, eigenvectors = np.linalg.eigh(R_matrix)
        
        # Extract the principal eigenvector (dominant spatial direction) -> shape (F, T, M)
        u1 = eigenvectors[..., -1]
        
        # 2. Calculate instantaneous Collinearity (rho)
        # Expand target_rtf for batched calculation -> shape (F, 1, M)
        d_exp = target_rtf[:, np.newaxis, :]
        
        # Batched dot product: |d^H * u1|
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1)) # shape: (F, T)
        
        # Normalize by vector magnitudes
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        den = d_norm * u1_norm + 1e-12
        
        rho = num / den # Instantaneous Collinearity mapped exactly to [0, 1]
        
        # 3. Continuous Transfer Function (Sigmoid)
        # Maps rho(t,f) -> zeta(t,f) smoothly to prevent phase chatter
        zeta_dynamic = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.kappa * (rho - self.rho_0)))
        
        return zeta_dynamic

    def process(self, R_matrix, target_rtf):
        """
        Executes the full DIAT pipeline.
        """
        F, T, _, _ = R_matrix.shape
        
        # Step 1: Identify the dynamic spatial regularization state
        zeta_tensor = self.identify(R_matrix, target_rtf)
        
        # Step 2: Set the mathematically optimal distortion penalty (from our heuristic sweep)
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        
        # Step 3: Treat the mixture
        weights = self.ulb_engine.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        
        return weights

    def apply_weights(self, stft_mix, weights):
        return self.ulb_engine.apply_weights(stft_mix, weights)
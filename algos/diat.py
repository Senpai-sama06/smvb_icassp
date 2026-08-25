import numpy as np
from algos.ulb import UniversalLinearBeamformer

class DIAT_Bf:
    """
    Dynamic Identify and Treat Beamformer (DIAT-Bf).
    Includes ablation modes for rigorous testing.
    """
    def __init__(self, num_mics=2, rho_0=0.85, kappa=20.0, zeta_min=1e-6, zeta_max=1e-2, ablation_mode=None):
        self.M = num_mics
        self.rho_0 = rho_0
        self.kappa = kappa
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ablation_mode = ablation_mode # None, 'binary', or 'static'
        
        self.ulb_engine = UniversalLinearBeamformer(num_mics=self.M)

    def identify(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        _, eigenvectors = np.linalg.eigh(R_matrix)
        u1 = eigenvectors[..., -1]
        
        d_exp = target_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        den = d_norm * u1_norm + 1e-12
        
        rho = num / den 
        
        # --- ABLATION ROUTINES ---
        if self.ablation_mode == 'static':
            # The Sledgehammer: Always use max regularization
            return np.full((F, T), self.zeta_max, dtype=np.float32)
            
        elif self.ablation_mode == 'binary':
            # The Old SMVB: Hard switch at the threshold
            zeta_dynamic = np.where(rho > self.rho_0, self.zeta_max, self.zeta_min)
            return zeta_dynamic
            
        else:
            # PROPOSED METHOD: The Continuous Sigmoid Clutch
            zeta_dynamic = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.kappa * (rho - self.rho_0)))
            return zeta_dynamic

    def process(self, R_matrix, target_rtf):
        F, T, _, _ = R_matrix.shape
        zeta_tensor = self.identify(R_matrix, target_rtf)
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        weights = self.ulb_engine.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        return weights

    def apply_weights(self, stft_mix, weights):
        return self.ulb_engine.apply_weights(stft_mix, weights)
import numpy as np
from algos.ulb import UniversalLinearBeamformer

class CDR_MVDR:
    """
    Continuous Dynamic Regularization (CDR) Layer for MVDR beamforming.
    Features Orthogonal Subspace Leakage with M-Scaled Sigmoid Clutch.
    """
    def __init__(self, num_mics=2, kappa=25.0, zeta_min=1e-5, zeta_max=0.5, ablation_mode=None):
        self.M = num_mics
        # Auto-scaling boundary based on array degrees of freedom
        self.epsilon_0 = 1.0 / self.M  
        self.kappa = kappa
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.ablation_mode = ablation_mode 
        self.ulb_engine = UniversalLinearBeamformer(num_mics=self.M)

    def compute_zeta(self, R_matrix, target_rtf):
        F, T, M, _ = R_matrix.shape
        if target_rtf.shape[0] == self.M and target_rtf.shape[1] != self.M:
            target_rtf = target_rtf.T
            
        _, eigenvectors = np.linalg.eigh(R_matrix)
        u1 = eigenvectors[..., -1]
        
        d_exp = target_rtf[:, np.newaxis, :]
        num = np.abs(np.sum(d_exp.conj() * u1, axis=-1))
        d_norm = np.linalg.norm(target_rtf, axis=-1)[:, np.newaxis]
        u1_norm = np.linalg.norm(u1, axis=-1)
        
        rho = np.clip(num / (d_norm * u1_norm + 1e-12), 0.0, 1.0)
        epsilon = 1.0 - (rho ** 2)
        trace_R = np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12
        
        # --- ABLATION ROUTINES ---
        if self.ablation_mode == 'nominal':
            z_factor = np.full((F, T), self.zeta_min, dtype=np.float32)

        elif self.ablation_mode == 'sledgehammer':
            z_factor = np.full((F, T), self.zeta_max, dtype=np.float32)
            
        elif self.ablation_mode == 'compromise':
            z_mid = 10 ** ((np.log10(self.zeta_min) + np.log10(self.zeta_max)) / 2)
            z_factor = np.full((F, T), z_mid, dtype=np.float32)
            
        elif self.ablation_mode == 'binary':
            z_factor = np.where(epsilon > self.epsilon_0, self.zeta_max, self.zeta_min)

        elif self.ablation_mode == 'linear':
            z_factor = self.zeta_min + (epsilon * self.zeta_max)
            
        else:
            # PROPOSED AUTO-CDR
            z_factor = self.zeta_min + (self.zeta_max - self.zeta_min) / (1.0 + np.exp(-self.kappa * (epsilon - self.epsilon_0)))

        return z_factor * trace_R

    def process(self, R_matrix, target_rtf):
        F, T, _, _ = R_matrix.shape
        zeta_tensor = self.compute_zeta(R_matrix, target_rtf)
        mu_inv_tensor = np.zeros((F, T), dtype=np.float32)
        weights = self.ulb_engine.process(R_matrix, target_rtf, mu_inv_tensor, zeta=zeta_tensor)
        return weights

    def apply_weights(self, stft_mix, weights):
        return self.ulb_engine.apply_weights(stft_mix, weights)
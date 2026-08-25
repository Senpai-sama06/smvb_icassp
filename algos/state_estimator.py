import numpy as np
from scipy.signal import lfilter

class SpatialStateEstimator:
    """
    The 'Identify' Module of the DIAT-Bf Architecture.
    Computes Kernel-Regularized Covariance and Continuous Spatial Entropy.
    Fully vectorized for offline/batch processing (Zero Python Loops).
    """
    def __init__(self, num_mics=2, alpha=0.9, mic_dist=0.08, fs=16000, n_fft=1024, kappa=0.1):
        self.M = num_mics
        self.alpha = alpha          # Causal EMA smoothing factor
        self.mic_dist = mic_dist    # Array spacing in meters
        self.fs = fs
        self.n_fft = n_fft
        self.kappa = kappa          # Regularization weight (how strongly to trust the diffuse prior)
        self.c = 343.0              # Speed of sound (m/s)
        
        # Pre-compute the Diffuse Spatial Coherence Matrix (Kernel Prior)
        self.Gamma = self._compute_diffuse_prior()

    def _compute_diffuse_prior(self):
        """
        Computes the theoretical diffuse coherence matrix Gamma(f) for all frequency bins.
        Using the spherical Bessel function (sinc).
        """
        F = self.n_fft // 2 + 1
        freqs = np.linspace(0, self.fs / 2, F)
        
        Gamma = np.zeros((F, self.M, self.M), dtype=complex)
        for i in range(self.M):
            for j in range(self.M):
                if i == j:
                    Gamma[:, i, j] = 1.0
                else:
                    # np.sinc in python is defined as sin(pi*x)/(pi*x)
                    # So we pass (2 * f * d / c) directly
                    arg = 2 * freqs * self.mic_dist / self.c
                    Gamma[:, i, j] = np.sinc(arg)
        return Gamma

    def process(self, stft_mix, target_rtf):
        """
        stft_mix: Complex numpy array of shape (M, F, T)
        target_rtf: Complex numpy array of shape (M, F) representing d(f)
        
        Returns:
            R_reg: Regularized Covariance Matrix (F, T, M, M)
            H: Continuous Spatial Entropy (F, T)
            rho: Continuous Collinearity (F, T)
        """
        M, F, T = stft_mix.shape
        
        # 1. Prepare Tensors
        y = stft_mix.transpose(1, 2, 0)[..., np.newaxis]          # (F, T, M, 1)
        y_conj = y.conj().transpose(0, 1, 3, 2)                   # (F, T, 1, M)
        d_f = target_rtf.T[:, np.newaxis, :, np.newaxis]          # (F, 1, M, 1)
        d_f_conj = d_f.conj().transpose(0, 1, 3, 2)               # (F, 1, 1, M)

        # 2. Empirical Covariance (IIR Filter)
        R_inst = np.matmul(y, y_conj)
        b, a = [1 - self.alpha], [1, -self.alpha]
        R_yy = lfilter(b, a, R_inst, axis=1)                      # (F, T, M, M)

        # 3. Kernel Regularization (The Physics Prior)
        # Calculate trace to match the energy of the empirical matrix
        trace_R = np.trace(R_yy, axis1=-2, axis2=-1).real         # (F, T)
        energy_scale = (trace_R / M)[..., np.newaxis, np.newaxis] # (F, T, 1, 1)
        
        # Expand Gamma to match time dimension: (F, 1, M, M)
        Gamma_expanded = self.Gamma[:, np.newaxis, :, :]
        
        # Blend empirical matrix with the scaled diffuse prior
        R_reg = (1 - self.kappa) * R_yy + self.kappa * energy_scale * Gamma_expanded

        # 4. Batched Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R_reg)
        
        # eigh sorts ascending. max is -1, min is 0
        lambda_M = eigenvalues[..., -1].real
        lambda_1 = eigenvalues[..., 0].real
        u_1 = eigenvectors[..., -1]                               # Principal eigenvector (F, T, M)

        # 5. Continuous Spatial Entropy (H)
        # Clip negative eigenvalues (numerical noise) to zero
        eig_sum = np.clip(lambda_M + lambda_1, 1e-12, None)
        p_M = np.clip(lambda_M / eig_sum, 1e-12, 1.0)
        p_1 = np.clip(lambda_1 / eig_sum, 1e-12, 1.0)
        
        # Von Neumann Entropy (Range: [0, 1] for M=2)
        H = - (p_M * np.log2(p_M) + p_1 * np.log2(p_1))

        # 6. Continuous Collinearity (rho)
        num = np.abs(np.einsum('ftmc,ftm->ft', d_f_conj, u_1))
        den = np.linalg.norm(d_f.squeeze(), axis=-1)[:, np.newaxis] * np.linalg.norm(u_1, axis=-1)
        rho = num / (den + 1e-12)

        return R_reg, H, rho
import numpy as np


class Evaluator:
    """
    Evaluation and spatial estimation utility class for acoustic signal processing
    and multi-channel beamforming.
    """

    def __init__(self, ref_mic=0):
        """
        Parameters:
            ref_mic: Reference microphone index (0-indexed, default: 0 for Mic 1).
        """
        self.ref_mic = ref_mic

    @staticmethod
    def calculate_si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
        """
        Computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
        between a reference signal and an estimated signal.

        Parameters:
            ref: Clean reference signal array (1D).
            est: Estimated/enhanced signal array (1D).

        Returns:
            si_sdr_db: SI-SDR value in decibels (dB).
        """
        ref = np.asarray(ref, dtype=np.float64).flatten()
        est = np.asarray(est, dtype=np.float64).flatten()

        # Ensure zero mean
        ref = ref - np.mean(ref)
        est = est - np.mean(est)

        # Calculate optimal scaling factor
        alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)

        # Project estimated signal onto reference
        target_scaled = alpha * ref
        noise = est - target_scaled

        # Calculate ratio in dB
        val = (np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise, noise) + 1e-12)
        return float(10 * np.log10(val))

    @staticmethod
    def get_oracle_rtf(target_stft: np.ndarray, ref_mic: int = 0) -> np.ndarray:
        """
        Computes the Oracle Relative Transfer Function (RTF) using the 
        clean target spatial covariance matrix (Principal Eigenvector method).

        Parameters:
            target_stft: Complex numpy array of shape (M, F, T)
            ref_mic: Reference microphone index (default: 0)

        Returns:
            rtf: Complex numpy array of shape (M, F) representing d(f)
        """
        M, F, T = target_stft.shape
        rtf = np.zeros((M, F), dtype=complex)

        for f in range(F):
            S = target_stft[:, f, :]
            # Target spatial covariance matrix (SCM)
            R_target = (S @ S.conj().T) / T

            # Extract principal eigenvector
            eigenvalues, eigenvectors = np.linalg.eigh(R_target)
            d_f = eigenvectors[:, -1]

            # Normalize to the reference microphone
            rtf[:, f] = d_f / (d_f[ref_mic] + 1e-12)

        return rtf

    def evaluate_performance(self, target: np.ndarray, mix: np.ndarray, enhanced: np.ndarray) -> dict:
        """
        Calculates input SI-SDR, output SI-SDR, and SDR improvement (delta).

        Parameters:
            target: Clean reference signal (shape: (M, Samples) or (Samples,)).
            mix: Unprocessed mixture signal (shape: (M, Samples) or (Samples,)).
            enhanced: Enhanced signal (shape: (Samples,)).

        Returns:
            dict containing 'sisdr_in', 'sisdr_out', and 'improvement' in dB.
        """
        ref_signal = target[self.ref_mic] if target.ndim > 1 else target
        unprocessed_signal = mix[self.ref_mic] if mix.ndim > 1 else mix
        enhanced_signal = enhanced.flatten()

        min_len = min(len(ref_signal), len(unprocessed_signal), len(enhanced_signal))
        ref_signal = ref_signal[:min_len]
        unprocessed_signal = unprocessed_signal[:min_len]
        enhanced_signal = enhanced_signal[:min_len]

        sisdr_in = self.calculate_si_sdr(ref_signal, unprocessed_signal)
        sisdr_out = self.calculate_si_sdr(ref_signal, enhanced_signal)
        improvement = sisdr_out - sisdr_in

        return {
            "sisdr_in": sisdr_in,
            "sisdr_out": sisdr_out,
            "improvement": improvement
        }


# --- Backward Compatibility Module-Level Functions ---
calculate_si_sdr = Evaluator.calculate_si_sdr
get_oracle_rtf = Evaluator.get_oracle_rtf

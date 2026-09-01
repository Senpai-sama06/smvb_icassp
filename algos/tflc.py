import numpy as np

def oracle_tflc_process(target_stft, interf_stft, mix_stft, target_rtf, interf_rtf, ref_mic=0):
    """
    Rigorous Oracle Time-Frequency Linearly Constrained (TFLC) Beamformer.
    Applies adaptive LCMV constraints using local RTFs.
    """
    M, F, T = target_stft.shape
    enhanced_stft = np.zeros((F, T), dtype=complex)
    
    # Target constraint (gain = 1), Interferer constraint (gain = 0)
    g = np.array([1.0, 0.0]).reshape(2, 1)
    
    # Dynamically check RTF shape to handle both (M, F) and (F, M) formats safely
    is_mf_shape = (target_rtf.shape[0] == M)
    
    for f in range(F):
        Y = mix_stft[:, f, :]
        
        # Batch covariance over the utterance
        R_y = (Y @ Y.conj().T) / T
        R_inv = np.linalg.pinv(R_y + 1e-6 * np.trace(R_y) * np.eye(M))
        
        # Extract constraints based on RTF array shape
        if is_mf_shape:
            c_t = target_rtf[:, f].reshape(M, 1)
            c_i = interf_rtf[:, f].reshape(M, 1)
        else:
            c_t = target_rtf[f, :].reshape(M, 1)
            c_i = interf_rtf[f, :].reshape(M, 1)
            
        C = np.hstack((c_t, c_i))  # Shape: (M, 2)
        
        # Standard LCMV closed-form solution: W = R^-1 C (C^H R^-1 C)^-1 g
        try:
            C_H_Rinv = C.conj().T @ R_inv
            inner_term = C_H_Rinv @ C
            inner_inv = np.linalg.pinv(inner_term + 1e-6 * np.eye(2))
            
            w_tflc = R_inv @ C @ inner_inv @ g
            w_tflc = w_tflc.flatten()
            
        except np.linalg.LinAlgError:
            # Fallback to MVDR if interferer RTF is highly collinear with target
            numerator = R_inv @ c_t
            denominator = (c_t.conj().T @ R_inv @ c_t)[0, 0]
            w_tflc = (numerator / (denominator + 1e-12)).flatten()
            
        enhanced_stft[f, :] = w_tflc.conj().T @ Y
            
    return enhanced_stft
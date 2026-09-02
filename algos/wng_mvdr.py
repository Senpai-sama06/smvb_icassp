import numpy as np

def get_wng_constrained_loading(R_matrix, d_vector, gamma_db, z_init_max=0.5, num_iter=25, max_expand_iter=15):
    """
    Vectorized bisection search with adaptive upper-bound expansion to find the 
    minimal diagonal loading required to achieve a target White Noise Gain (gamma_db).
    """
    F, T, M, _ = R_matrix.shape
    gamma_linear = 10 ** (-gamma_db / 10.0) 

    if d_vector.ndim == 2:
        d_exp = d_vector[:, np.newaxis, :, np.newaxis]
    else:
        d_exp = d_vector[..., np.newaxis]
        
    d_conj = d_exp.conj().transpose(0, 1, 3, 2)
    trace_power = (np.trace(R_matrix, axis1=2, axis2=3).real + 1e-12) / M
    I_matrix = np.eye(M)[np.newaxis, np.newaxis, :, :]

    # --- PHASE 1: Adaptive Upper-Bound Expansion ---
    z_high = np.full((F, T), z_init_max, dtype=np.float64)
    
    for _ in range(max_expand_iter):
        R_loaded = R_matrix + (z_high * trace_power)[..., np.newaxis, np.newaxis] * I_matrix
        
        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)
            
        num = np.matmul(R_inv, d_exp)
        den = np.matmul(d_conj, num) + 1e-12
        w = num / den
        w_norm_sq = np.sum(np.abs(w)**2, axis=(2, 3))
        
        # Check which TF bins still violate the constraint (WNG too low / norm too high)
        infeasible = w_norm_sq > gamma_linear
        
        if not np.any(infeasible):
            break
            
        # Geometrically expand the upper bound for the infeasible bins
        z_high = np.where(infeasible, z_high * 2.0, z_high)
        
    if np.any(w_norm_sq > gamma_linear):
        print("WARNING: WNG constraint not reachable for some bins within expansion limit.")

    # --- PHASE 2: Tighter Bisection Search (25 iterations) ---
    z_low = np.zeros((F, T), dtype=np.float64)
    
    for _ in range(num_iter):
        z_mid = (z_low + z_high) / 2.0
        R_loaded = R_matrix + (z_mid * trace_power)[..., np.newaxis, np.newaxis] * I_matrix
        
        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)
            
        num = np.matmul(R_inv, d_exp)
        den = np.matmul(d_conj, num) + 1e-12
        w = num / den
        w_norm_sq = np.sum(np.abs(w)**2, axis=(2, 3))
        
        mask_needs_more_loading = w_norm_sq > gamma_linear
        
        z_low = np.where(mask_needs_more_loading, z_mid, z_low)
        z_high = np.where(~mask_needs_more_loading, z_mid, z_high)

    return z_high
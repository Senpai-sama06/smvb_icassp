import os
import sys
import random
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pathlib import Path
from scipy.signal import lfilter

from pesq import pesq
from pystoi import stoi

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer 
from algos.v_smvb import SMVB
from algos.diat import DIAT_Bf

# --- NEW IMPORTS ---
from algos.gevd import oracle_gevd_process
from algos.tflc import oracle_tflc_process

def calculate_all_metrics(target, est, mix, fs=16000):
    """Calculates SI-SDR Improvement, PESQ, and STOI."""
    min_len = min(len(target), len(est), len(mix))
    t, e, m = target[:min_len], est[:min_len], mix[:min_len]

    def si_sdr(ref, sig):
        ref = ref - np.mean(ref); sig = sig - np.mean(sig)
        alpha = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-12)
        target_scaled = alpha * ref
        noise_sig = sig - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise_sig, noise_sig) + 1e-12))
    
    sisdr_imp = si_sdr(t, e) - si_sdr(t, m)
    try: p_score = pesq(fs, t, e, 'wb')
    except: p_score = 1.0 
    s_score = stoi(t, e, fs, extended=False)

    return {'SI-SDR_Imp': sisdr_imp, 'PESQ': p_score, 'STOI': s_score}

def evaluate_bss_perceptual(bss_function, X_bss, target, mix, hop_length, fs, **kwargs):
    """Safely executes BSS with correct Time/Freq axes."""
    try:
        Y_bss = bss_function(X_bss, **kwargs) 
        if Y_bss.ndim == 3: 
            Y_bss_stft = Y_bss.transpose(2, 1, 0)
        else: 
            return None
        
        best_metrics = {'SI-SDR_Imp': -999.0, 'PESQ': -999.0, 'STOI': -999.0}
        for i in range(Y_bss_stft.shape[0]):
            est_time = librosa.istft(Y_bss_stft[i], hop_length=hop_length)
            metrics = calculate_all_metrics(target, est_time, mix, fs)
            if metrics['SI-SDR_Imp'] > best_metrics['SI-SDR_Imp']:
                best_metrics = metrics
        return best_metrics
    except Exception as e: 
        print(f"      [!] BSS Error: {e}")
        return None

def oracle_mwf_process(target_stft, mix_stft, ref_mic=0):
    """Mathematically pristine Oracle Multichannel Wiener Filter."""
    M, F, T = target_stft.shape
    enhanced_stft = np.zeros((F, T), dtype=complex)
    for f in range(F):
        S = target_stft[:, f, :]
        Y = mix_stft[:, f, :]
        R_x = (S @ S.conj().T) / T
        R_y = (Y @ Y.conj().T) / T
        W = np.linalg.pinv(R_y + 1e-6 * np.eye(M)) @ R_x
        w = W[:, ref_mic]
        enhanced_stft[f, :] = w.conj().T @ Y
    return enhanced_stft

def main():
    print("--- GENERATING FULL PERCEPTUAL EVALUATION ---")
    fs = 16000
    simulator = AcousticSceneSimulator(snr_target_db=25, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    n_fft = 1024; hop_length = 256
    angles = [30, 45, 60, 75, 80, 85, 88]
    
    results_sisdr, results_pesq, results_stoi = [], [], []

    for angle in angles:
        print(f"Simulating Angle: {angle}°")
        random.seed(42); np.random.seed(42) 
        
        mix, target, interferer, noise = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[angle], save_outputs=False)
        
        # Ensure all components are properly transposed to (channels, time)
        mix = mix.T; target = target.T; interferer = interferer.T; noise = noise.T
        
        # Generate all STFTs
        mix_stft = np.stack([librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)])
        target_stft = np.stack([librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)])
        interf_stft = np.stack([librosa.stft(interferer[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(interferer[1], n_fft=n_fft, hop_length=hop_length)])
        noise_stft = np.stack([librosa.stft(noise[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(noise[1], n_fft=n_fft, hop_length=hop_length)])
        
        # Calculate Target and Interferer RTFs
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        interf_rtf = evaluator.get_oracle_rtf(interf_stft, ref_mic=0)
        
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.7], [1, -0.7], np.matmul(y, y_conj), axis=1)

        F, T = mix_stft.shape[1], mix_stft.shape[2]
        
        row_sisdr, row_pesq, row_stoi = {'Angle': angle}, {'Angle': angle}, {'Angle': angle}
        
        def store(name, metrics):
            if metrics:
                row_sisdr[name] = metrics['SI-SDR_Imp']
                row_pesq[name] = metrics['PESQ']
                row_stoi[name] = metrics['STOI']

        # 1. Blind Benchmarks
        X_bss = mix_stft.transpose(2, 1, 0)
        if hasattr(pra.bss, 'auxiva'):
            store('Blind: AuxIVA', evaluate_bss_perceptual(pra.bss.auxiva, X_bss, target[0], mix[0], hop_length, fs, n_iter=30))
        if hasattr(pra.bss, 'ilrma'):
            store('Blind: ILRMA', evaluate_bss_perceptual(pra.bss.ilrma, X_bss, target[0], mix[0], hop_length, fs, n_iter=30, n_components=2))  

        # 2. Oracle Classical Baselines
        w_mvdr = UniversalLinearBeamformer(num_mics=2).process(R_matrix, oracle_rtf, np.zeros((F,T)), zeta=1e-6)
        store('Oracle: MVDR', calculate_all_metrics(target[0], librosa.istft(UniversalLinearBeamformer().apply_weights(mix_stft, w_mvdr), hop_length=hop_length), mix[0], fs))
        
        est_mwf = librosa.istft(oracle_mwf_process(target_stft, mix_stft), hop_length=hop_length)
        store('Oracle: MWF (Max-SNR)', calculate_all_metrics(target[0], est_mwf, mix[0], fs))

        # --- 3. NEW MODERN BASELINES ---
        est_gevd = librosa.istft(oracle_gevd_process(target_stft, noise_stft, mix_stft, ref_mic=0), hop_length=hop_length)
        store('Oracle: GEVD', calculate_all_metrics(target[0], est_gevd, mix[0], fs))
        
        est_tflc = librosa.istft(oracle_tflc_process(target_stft, interf_stft, mix_stft, oracle_rtf, interf_rtf, ref_mic=0), hop_length=hop_length)
        store('Oracle: TFLC', calculate_all_metrics(target[0], est_tflc, mix[0], fs))

        # 4. Prior Work (SMVB)
        smvb = SMVB(num_mics=2)
        store('Prior: Old SMVB', calculate_all_metrics(target[0], librosa.istft(smvb.process(mix_stft, oracle_rtf), hop_length=hop_length), mix[0], fs))
        
        # 5. Proposed (DIAT-Bf)
        diat = DIAT_Bf(num_mics=2, zeta_min=1e-4) 
        w_diat = diat.process(R_matrix, oracle_rtf)
        store('Proposed: DIAT-Bf', calculate_all_metrics(target[0], librosa.istft(diat.apply_weights(mix_stft, w_diat), hop_length=hop_length), mix[0], fs))
            
        results_sisdr.append(row_sisdr); results_pesq.append(row_pesq); results_stoi.append(row_stoi)

    df_sisdr = pd.DataFrame(results_sisdr).set_index('Angle')
    df_pesq = pd.DataFrame(results_pesq).set_index('Angle')
    df_stoi = pd.DataFrame(results_stoi).set_index('Angle')
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [(df_sisdr, 'SI-SDR Improvement (dB)', axs[0]), (df_pesq, 'PESQ (Wideband)', axs[1]), (df_stoi, 'STOI', axs[2])]
    
    for df, title, ax in metrics:
        if 'Blind: AuxIVA' in df: ax.plot(df.index, df['Blind: AuxIVA'], label='Blind: AuxIVA', linestyle='--', color='gray', marker='o')
        if 'Blind: ILRMA' in df: ax.plot(df.index, df['Blind: ILRMA'], label='Blind: ILRMA', linestyle='--', color='silver', marker='x')
        
        ax.plot(df.index, df['Oracle: MVDR'], label='Oracle: MVDR', linestyle='-', color='sienna', marker='s')
        ax.plot(df.index, df['Oracle: MWF (Max-SNR)'], label='Oracle: MWF', linestyle='-', color='orange', marker='v')
        
        # Plot new baselines
        if 'Oracle: GEVD' in df: ax.plot(df.index, df['Oracle: GEVD'], label='Oracle: GEVD', linestyle='-.', color='purple', marker='p')
        if 'Oracle: TFLC' in df: ax.plot(df.index, df['Oracle: TFLC'], label='Oracle: TFLC', linestyle='-.', color='forestgreen', marker='*')
        
        ax.plot(df.index, df['Prior: Old SMVB'], label='Prior: Old SMVB', linestyle='-', color='steelblue', marker='^')
        ax.plot(df.index, df['Proposed: DIAT-Bf'], label='Proposed: DIAT-Bf', linewidth=3.5, linestyle='-', color='crimson', marker='D', zorder=10)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Interferer Angle (Degrees)', fontsize=12)
        ax.set_xticks(angles)
        ax.grid(True, linestyle='--', alpha=0.6)
        if ax == axs[0]: ax.legend(loc="best", fontsize=9) 

    plt.tight_layout()
    plt.savefig('fair_perceptual_plot.png', dpi=300)
    plt.show()
    print("\nSaved 'fair_perceptual_plot.png'")

if __name__ == "__main__":
    main()
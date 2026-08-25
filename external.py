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

# Import Perceptual Metrics
from pesq import pesq
from pystoi import stoi

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator
from algos.ulb import UniversalLinearBeamformer
from algos.v_smvb import SMVB
from algos.diat import DIAT_Bf

def calculate_all_metrics(target, est, mix, fs=16000):
    """Calculates SI-SDR Improvement, PESQ, and STOI."""
    # Ensure lengths match exactly
    min_len = min(len(target), len(est), len(mix))
    t = target[:min_len]
    e = est[:min_len]
    m = mix[:min_len]

    # 1. SI-SDR Calculation
    def si_sdr(ref, sig):
        ref = ref - np.mean(ref); sig = sig - np.mean(sig)
        alpha = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-12)
        target_scaled = alpha * ref
        noise = sig - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-12) / (np.dot(noise, noise) + 1e-12))
    
    sisdr_imp = si_sdr(t, e) - si_sdr(t, m)
    
    # 2. PESQ (Wideband requires 16kHz)
    try:
        p_score = pesq(fs, t, e, 'wb')
    except:
        p_score = 1.0 # Fallback if audio is completely silent/destroyed

    # 3. STOI
    s_score = stoi(t, e, fs, extended=False)

    return {'SI-SDR_Imp': sisdr_imp, 'PESQ': p_score, 'STOI': s_score}

def evaluate_bss_perceptual(bss_function, X_bss, target, mix, hop_length, fs, **kwargs):
    try:
        Y_bss = bss_function(X_bss, **kwargs)
        if Y_bss.ndim == 3: Y_bss_stft = Y_bss.transpose(2, 0, 1)
        else: return None
        
        best_metrics = {'SI-SDR_Imp': -999.0, 'PESQ': -999.0, 'STOI': -999.0}
        for i in range(Y_bss_stft.shape[0]):
            est_time = librosa.istft(Y_bss_stft[i], hop_length=hop_length)
            metrics = calculate_all_metrics(target, est_time, mix, fs)
            if metrics['SI-SDR_Imp'] > best_metrics['SI-SDR_Imp']:
                best_metrics = metrics
        return best_metrics
    except: return None

def main():
    print("--- GENERATING PERCEPTUAL EVALUATION GRAPHS ---")
    fs = 16000
    simulator = AcousticSceneSimulator(snr_target_db=25, fs=fs)
    evaluator = Evaluator(ref_mic=0)
    n_fft = 1024; hop_length = 256
    angles = [30, 45, 60, 75, 80, 85, 88]
    
    results_sisdr = []; results_pesq = []; results_stoi = []

    for angle in angles:
        print(f"Simulating Angle: {angle}°")
        random.seed(42); np.random.seed(42) # Lock dataset
        
        mix, target, interferer, noise = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[angle], save_outputs=False)
        mix = mix.T; target = target.T
        
        mix_stft = np.stack([librosa.stft(mix[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(mix[1], n_fft=n_fft, hop_length=hop_length)])
        target_stft = np.stack([librosa.stft(target[0], n_fft=n_fft, hop_length=hop_length), librosa.stft(target[1], n_fft=n_fft, hop_length=hop_length)])
        
        oracle_rtf = evaluator.get_oracle_rtf(target_stft, ref_mic=0)
        y = mix_stft.transpose(1, 2, 0)[..., np.newaxis]
        y_conj = y.conj().transpose(0, 1, 3, 2)
        R_matrix = lfilter([1 - 0.7], [1, -0.7], np.matmul(y, y_conj), axis=1)

        F, T = mix_stft.shape[1], mix_stft.shape[2]
        
        row_sisdr = {'Angle': angle}; row_pesq = {'Angle': angle}; row_stoi = {'Angle': angle}
        
        def store_metrics(name, metrics):
            if metrics:
                row_sisdr[name] = metrics['SI-SDR_Imp']
                row_pesq[name] = metrics['PESQ']
                row_stoi[name] = metrics['STOI']

        # 1. Blind Benchmark (AuxIVA)
        if hasattr(pra.bss, 'auxiva'):
            store_metrics('Blind: AuxIVA', evaluate_bss_perceptual(pra.bss.auxiva, mix_stft.transpose(1,2,0), target[0], mix[0], hop_length, fs, n_iter=30))
            
        # 2. Oracle MVDR (WNG Victim)
        ulb = UniversalLinearBeamformer(num_mics=2)
        w_mvdr = ulb.process(R_matrix, oracle_rtf, np.zeros((F,T)), zeta=1e-6)
        store_metrics('Oracle: Standard MVDR', calculate_all_metrics(target[0], librosa.istft(ulb.apply_weights(mix_stft, w_mvdr), hop_length=hop_length), mix[0], fs))
        
        # 3. Prior Work (SMVB)
        smvb = SMVB(num_mics=2)
        store_metrics('Prior: Old SMVB', calculate_all_metrics(target[0], librosa.istft(smvb.process(mix_stft, oracle_rtf), hop_length=hop_length), mix[0], fs))
        
        # 4. Proposed (DIAT-Bf with zeta_min=1e-4)
        diat = DIAT_Bf(num_mics=2, zeta_min=1e-4) 
        w_diat = diat.process(R_matrix, oracle_rtf)
        store_metrics('Proposed: DIAT-Bf', calculate_all_metrics(target[0], librosa.istft(diat.apply_weights(mix_stft, w_diat), hop_length=hop_length), mix[0], fs))
            
        results_sisdr.append(row_sisdr)
        results_pesq.append(row_pesq)
        results_stoi.append(row_stoi)

    df_sisdr = pd.DataFrame(results_sisdr).set_index('Angle')
    df_pesq = pd.DataFrame(results_pesq).set_index('Angle')
    df_stoi = pd.DataFrame(results_stoi).set_index('Angle')
    
    # --- PLOTTING 1x3 SUBPLOTS ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        (df_sisdr, 'SI-SDR Improvement (dB)', axs[0]),
        (df_pesq, 'PESQ (Wideband)', axs[1]),
        (df_stoi, 'STOI', axs[2])
    ]
    
    for df, title, ax in metrics:
        if 'Blind: AuxIVA' in df: ax.plot(df.index, df['Blind: AuxIVA'], label='Blind: AuxIVA', linestyle='--', color='gray', marker='o')
        ax.plot(df.index, df['Oracle: Standard MVDR'], label='Oracle: MVDR', linestyle='-', color='sienna', marker='s')
        ax.plot(df.index, df['Prior: Old SMVB'], label='Prior: Old SMVB', linestyle='-', color='steelblue', marker='^')
        ax.plot(df.index, df['Proposed: DIAT-Bf'], label='Proposed: DIAT-Bf', linewidth=3.5, linestyle='-', color='crimson', marker='D', zorder=10)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Interferer Angle (Degrees)', fontsize=12)
        ax.set_xticks(angles)
        ax.grid(True, linestyle='--', alpha=0.6)
        if ax == axs[0]: ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig('perceptual_metrics_plot.png', dpi=300)
    print("\nSaved 'perceptual_metrics_plot.png'")

if __name__ == "__main__":
    main()
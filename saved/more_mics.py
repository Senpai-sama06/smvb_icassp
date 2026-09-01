import os
import sys
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pathlib import Path
from scipy.signal import fftconvolve

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

from util.simulator import AcousticSceneSimulator
from util.evaluator import Evaluator

def get_circular_array(num_mics, radius=0.04, center=[2.45, 2.45, 1.5]):
    """Generates a circular microphone array with a given radius."""
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
    locs = np.zeros((3, num_mics))
    locs[0, :] = center[0] + radius * np.cos(angles)
    locs[1, :] = center[1] + radius * np.sin(angles)
    locs[2, :] = center[2]
    return locs

# --- BULLETPROOF SIMULATOR FOR M MICROPHONES ---
def patched_simulate(self, n=1, reverb=True, target_rt60=0.5, interferer_angles=None, save_outputs=False):
    M = self.mic_locs.shape[1]
    
    target_sig, interferer_sigs = self.load_audio_sources(n=n)
    room = self.setup_room(reverb=reverb, target_rt60=target_rt60)
    self.place_sources(room, n=n, interferer_angles=interferer_angles)
    room.compute_rir()
    
    def get_convolved(sig, rir):
        return fftconvolve(sig, rir, mode='full')
        
    def match_length(sig, target_length):
        if len(sig) > target_length: return sig[:target_length]
        elif len(sig) < target_length: return np.pad(sig, (0, target_length - len(sig)), 'constant')
        return sig

    # Compute raw target across all M mics
    target_raw = []
    for m in range(M):
        target_raw.append(get_convolved(target_sig, room.rir[m][0]))
        
    final_len = len(target_raw[0])
    
    # FIX: Force all target channels to be frame-perfect
    target_channels = [match_length(target_raw[m], final_len) for m in range(M)]
    
    # Compute interferers across all M mics
    interf_channels_total = [np.zeros(final_len) for _ in range(M)]
    
    if n > 0:
        for i, i_sig in enumerate(interferer_sigs):
            src_idx = i + 1
            for m in range(M):
                i_ch = get_convolved(i_sig, room.rir[m][src_idx])
                interf_channels_total[m] += match_length(i_ch, final_len)

    # Scale interferers to hit target SIR
    p_target = np.mean(target_channels[0] ** 2)
    p_interf = np.mean(interf_channels_total[0] ** 2)
    
    if p_interf > 0:
        scaling_factor = np.sqrt(p_target / (p_interf * (10 ** (self.sir_target_db / 10))))
        for m in range(M):
            interf_channels_total[m] *= scaling_factor

    # Mix and add noise
    final_mixes, final_noises = [], []
    for m in range(M):
        clean_mix = target_channels[m] + interf_channels_total[m]
        final_ch, noise_ch = self.add_awgn(clean_mix, self.snr_target_db)
        final_mixes.append(final_ch)
        final_noises.append(noise_ch)

    # Stack into arrays (Shape: M x Samples)
    stereo_mix = np.stack(final_mixes, axis=0).T
    stereo_target = np.stack(target_channels, axis=0).T
    stereo_interf = np.stack(interf_channels_total, axis=0).T
    stereo_noise = np.stack(final_noises, axis=0).T
    
    peak = np.max(np.abs(stereo_mix)) + 1e-9
    stereo_mix /= peak; stereo_target /= peak; stereo_interf /= peak; stereo_noise /= peak

    return stereo_mix, stereo_target, stereo_interf, stereo_noise

# Inject the patch into the class
AcousticSceneSimulator.simulate = patched_simulate


def main():
    print("--- RUNNING BSS MICROPHONE SCALING EXPERIMENT ---")
    fs = 16000
    n_fft = 1024; hop_length = 256
    angle = 45 
    
    mic_counts = [2, 3, 4, 6, 8]
    bss_sisdr_results = []

    for M in mic_counts:
        print(f"\nTesting AuxIVA with {M} Microphones...")
        random.seed(42); np.random.seed(42)
        
        mic_locs = get_circular_array(M)
        simulator = AcousticSceneSimulator(snr_target_db=25, fs=fs, mic_locs=mic_locs)
        
        mix, target, _, _ = simulator.simulate(n=1, reverb=True, target_rt60=0.2, interferer_angles=[angle], save_outputs=False)
        mix = mix.T; target = target.T
        
        mix_stft = np.stack([librosa.stft(mix[i], n_fft=n_fft, hop_length=hop_length) for i in range(M)])
        
        # PyRoomAcoustics expects (Time, Freq, Channels)
        X_bss = mix_stft.transpose(2, 1, 0)
        
        try:
            # FIX: Removed manual projection_back; AuxIVA handles it automatically.
            Y_bss_stft = pra.bss.auxiva(X_bss, n_iter=30).transpose(2, 1, 0)
            print("transposition")
            
            evaluator = Evaluator(ref_mic=0)
            best_imp = -999.0
            
            # Since n=1 (1 interferer + 1 target), AuxIVA outputs M components if M > 2.
            # We search all outputs to find the one representing the target.
            num_sources = Y_bss_stft.shape[0]
            for i in range(num_sources):
                est_time = librosa.istft(Y_bss_stft[i], hop_length=hop_length)
                res = evaluator.evaluate_performance(target[0], mix[0], est_time)
                if res['improvement'] > best_imp:
                    best_imp = res['improvement']
            
            bss_sisdr_results.append(best_imp)
            print(f"  -> Best SI-SDR Improvement: {best_imp:+.2f} dB")
        except Exception as e:
            print(f"  -> Failed: {e}")
            bss_sisdr_results.append(None)

    plt.figure(figsize=(8, 5))
    plt.plot(mic_counts, bss_sisdr_results, marker='o', linestyle='-', color='teal', linewidth=2.5)
    plt.title('Blind Source Separation (AuxIVA) vs. Microphone Count', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Microphones (M)', fontsize=12)
    plt.ylabel('SI-SDR Improvement (dB)', fontsize=12)
    plt.xticks(mic_counts)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('bss_mic_scaling.png', dpi=300)
    print("\nSaved plot to 'bss_mic_scaling.png'.")

if __name__ == "__main__":
    main()
import numpy as np
import soundfile as sf
import os
import glob
import random
import argparse
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import json


class AudioDatasetFetcher:
    """Handles fetching and discovery of audio datasets from Kaggle / local paths."""

    @staticmethod
    def get_audio_files(dataset_name, n_needed, min_duration=5.0):
        print(f"--- Fetching {n_needed} files (>= {min_duration}s) from: {dataset_name} ---")
        files = []
        try:
            if dataset_name == 'librispeech':
                path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
                files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
            elif dataset_name == 'musan':
                path = kagglehub.dataset_download("dogrose/musan-dataset")
                files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
            else: 
                path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
                wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
                files = glob.glob(os.path.join(wav_path, "*.wav"))
            
            if len(files) == 0: 
                raise ValueError(f"No files found for {dataset_name}")

            random.shuffle(files)
            valid_files = []
            
            for f in files:
                if len(valid_files) >= n_needed:
                    break
                try:
                    info = sf.info(f)
                    if info.duration >= min_duration:
                        valid_files.append(f)
                except Exception:
                    continue

            if len(valid_files) < n_needed:
                while len(valid_files) < n_needed:
                    valid_files += valid_files
                valid_files = valid_files[:n_needed]
            
            return valid_files
        except Exception as e:
            print(f"Error getting data: {e}")
            return []

    @staticmethod
    def get_single_dataset_files(dataset_name, n_needed, min_duration=5.0):
        """Fetches n_needed files from a specific dataset."""
        print(f"--- Fetching {n_needed} files (>= {min_duration}s) from: {dataset_name} ---")
        files = []
        
        if dataset_name == 'librispeech':
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan_speech':
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            # Explicitly grabbing speech from MUSAN to avoid pure noise/music for now
            files = glob.glob(os.path.join(path, "musan", "speech", "**", "*.wav"), recursive=True)
        elif dataset_name == 'ljspeech': 
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))
        
        if not files: 
            raise ValueError(f"No files found for {dataset_name}")

        random.shuffle(files)
        valid_files = []
        
        for f in files:
            if len(valid_files) >= n_needed:
                break
            try:
                if sf.info(f).duration >= min_duration:
                    valid_files.append(f)
            except Exception:
                continue

        # Duplicate if we run out of unique files meeting the length requirement
        if len(valid_files) < n_needed:
            while len(valid_files) < n_needed:
                valid_files += valid_files
            valid_files = valid_files[:n_needed]
        
        return valid_files


class AcousticSceneSimulator:
    """
    Object-oriented simulator for shoebox room acoustics, spatial audio mixtures,
    reverberant impulse responses (RIRs), interferer placement, and sensor noise.
    """

    # Default Room and Simulation Constants
    DEFAULT_FS = 16000
    DEFAULT_ROOM_DIM = [4.9, 4.9, 4.9]
    DEFAULT_RT60 = 0.5
    DEFAULT_SNR_TARGET_DB = 60  # Set to 60dB (imperceptible) to act purely as mathematical matrix regularization
    DEFAULT_SIR_TARGET_DB = 0

    # Default Mic Array (Center of room, 8cm spacing)
    DEFAULT_MIC_LOCS = np.array([
        [2.41, 2.45, 1.5],  # Mic 1 (Left)
        [2.49, 2.45, 1.5]   # Mic 2 (Right)
    ]).T

    DEFAULT_RADIUS = 2.0
    DEFAULT_MIN_SRC_DIST = 0.5

    def __init__(
        self,
        fs=DEFAULT_FS,
        room_dim=None,
        default_rt60=DEFAULT_RT60,
        snr_target_db=DEFAULT_SNR_TARGET_DB,
        sir_target_db=DEFAULT_SIR_TARGET_DB,
        mic_locs=None,
        radius=DEFAULT_RADIUS,
        min_src_dist=DEFAULT_MIN_SRC_DIST,
        dataset_fetcher=None
    ):
        self.fs = fs
        self.room_dim = room_dim if room_dim is not None else list(self.DEFAULT_ROOM_DIM)
        self.default_rt60 = default_rt60
        self.snr_target_db = snr_target_db
        self.sir_target_db = sir_target_db
        self.mic_locs = mic_locs if mic_locs is not None else np.copy(self.DEFAULT_MIC_LOCS)
        self.radius = radius
        self.min_src_dist = min_src_dist
        self.dataset_fetcher = dataset_fetcher if dataset_fetcher is not None else AudioDatasetFetcher()
        self.room = None

    @staticmethod
    def add_awgn(signal, snr_db):
        """
        Adds mathematically negligible sensor noise (e.g. 60dB SNR).
        This acts as a regularizer to prevent singular matrix inversion errors 
        in DSP algorithms without audibly degrading the signal.
        """
        sig_power = np.mean(signal ** 2)
        if sig_power == 0:
            return signal, np.zeros(signal.shape)
        
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        return signal + noise, noise

    def load_audio_sources(self, n=1):
        """Loads and zero-pads target (LJSpeech) and interferers (LibriSpeech/MUSAN)."""
        print("\n" + "=" * 40)
        print("LOADING DISTINCT DATASETS...")
        
        # Target is ALWAYS LJSpeech
        target_files = self.dataset_fetcher.get_single_dataset_files('ljspeech', 1)
        
        # Interferers are drawn from LibriSpeech (and MUSAN if >1 interferer)
        interferer_files = []
        if n > 0:
            interferer_files += self.dataset_fetcher.get_single_dataset_files('librispeech', max(1, n - 1))
        if n > 1:
            interferer_files += self.dataset_fetcher.get_single_dataset_files('musan_speech', 1)
            
        files = target_files + interferer_files[:n]

        print("\nSELECTED AUDIO SOURCES:")
        print(f"  [TARGET (LJSpeech)]   : {os.path.basename(files[0])}")
        for i, f_path in enumerate(files[1:]):
            print(f"  [INTERFERER {i+1}]      : {os.path.basename(f_path)}")
        print("=" * 40 + "\n")

        sigs = []
        max_len = 0
        
        for f in files:
            y, _ = librosa.load(f, sr=self.fs, mono=True)
            sigs.append(y)
            if len(y) > max_len:
                max_len = len(y)
        
        # Pad all signals with zeros to match the longest signal length
        sigs = [np.pad(s, (0, max_len - len(s)), 'constant') for s in sigs]
        target_sig = sigs[0]
        interferer_sigs = sigs[1:]
        return target_sig, interferer_sigs

    def setup_room(self, reverb=True, target_rt60=None):
        """Initializes ShoeBox room and adds microphone array."""
        rt60 = target_rt60 if target_rt60 is not None else self.default_rt60
        if reverb:
            e_absorption, max_order = pra.inverse_sabine(rt60, self.room_dim)
            materials = pra.Material(e_absorption)
            m_order = max_order 
        else:
            materials = pra.Material(1.0)
            m_order = 0

        room = pra.ShoeBox(self.room_dim, fs=self.fs, materials=materials, max_order=m_order)
        room.add_microphone_array(self.mic_locs)
        self.room = room
        return room

    def place_sources(self, room, n=1, interferer_angles=None):
        """Places target and interferer sources in equidistant circular placement."""
        mic_center = np.mean(self.mic_locs, axis=1) 
        placed_sources = []

        target_angle = np.pi / 2 
        pos_target = [
            mic_center[0] + self.radius * np.cos(target_angle),
            mic_center[1] + self.radius * np.sin(target_angle),
            1.5
        ]
        room.add_source(pos_target)
        placed_sources.append(np.array(pos_target))

        if n > 0:
            for i in range(n):
                max_attempts = 100
                for attempt in range(max_attempts):
                    # If specific angles are provided, bypass the collision check
                    # so we can explicitly test encroachment.
                    if interferer_angles is not None and len(interferer_angles) > i:
                        theta = interferer_angles[i] * (np.pi / 180)
                        ignore_collision = True
                    else:
                        theta = random.uniform(0, 2 * np.pi)
                        ignore_collision = False

                    cand_x = mic_center[0] + self.radius * np.cos(theta)
                    cand_y = mic_center[1] + self.radius * np.sin(theta)
                    candidate = np.array([cand_x, cand_y, 1.5])
                    
                    if ignore_collision:
                        collision = False
                    else:
                        collision = any(np.linalg.norm(candidate - es) < self.min_src_dist for es in placed_sources)
                        
                    if not collision:
                        room.add_source(candidate)
                        placed_sources.append(candidate)
                        break
                
                # Failsafe check
                if len(placed_sources) <= i + 1:
                    raise RuntimeError(f"Could not place interferer {i+1} without collision.")
        return placed_sources

    def simulate(self, n=1, reverb=True, target_rt60=0.5, interferer_angles=None, save_outputs=False):
        """Simulates the entire acoustic scene with target, interferers, room RIRs, and noise."""
        # --- 1. Load Audio (Distinct Datasets) ---
        target_sig, interferer_sigs = self.load_audio_sources(n=n)

        # --- 2. Setup Room ---
        room = self.setup_room(reverb=reverb, target_rt60=target_rt60)

        # --- 3. Add Sources (Equidistant Circular Placement) ---
        self.place_sources(room, n=n, interferer_angles=interferer_angles)

        # --- 5. Compute RIRs ---
        room.compute_rir()
        
        # --- 7. Convolution & Mixing ---
        def get_convolved(sig, rir):
            return fftconvolve(sig, rir, mode='full')
            
        def match_length(sig, target_length):
            """Truncates or zero-pads a signal to perfectly match target_length."""
            if len(sig) > target_length:
                return sig[:target_length]
            elif len(sig) < target_length:
                return np.pad(sig, (0, target_length - len(sig)), 'constant')
            return sig

        target_ch1 = get_convolved(target_sig, room.rir[0][0])
        target_ch2 = get_convolved(target_sig, room.rir[1][0])

        # Initialize interferer accumulators based on the convolved target length
        final_len = len(target_ch1)
        interf_ch1_total = np.zeros(final_len)
        interf_ch2_total = np.zeros(final_len)

        if n > 0:
            for i, i_sig in enumerate(interferer_sigs):
                src_idx = i + 1
                i_ch1 = get_convolved(i_sig, room.rir[0][src_idx])
                i_ch2 = get_convolved(i_sig, room.rir[1][src_idx])
                
                # Apply match_length to prevent broadcasting ValueError
                interf_ch1_total += match_length(i_ch1, final_len)
                interf_ch2_total += match_length(i_ch2, final_len)

        # FIX: Exact SIR Scaling
        p_target = np.mean(target_ch1 ** 2)
        p_interf = np.mean(interf_ch1_total ** 2)
        
        if p_interf > 0:
            scaling_factor = np.sqrt(p_target / (p_interf * (10 ** (self.sir_target_db / 10))))
            interf_ch1_total *= scaling_factor
            interf_ch2_total *= scaling_factor

        clean_mix_ch1 = target_ch1 + interf_ch1_total
        clean_mix_ch2 = target_ch2 + interf_ch2_total

        final_ch1, noise_ch1 = self.add_awgn(clean_mix_ch1, self.snr_target_db)
        final_ch2, noise_ch2 = self.add_awgn(clean_mix_ch2, self.snr_target_db)

        # --- 8. Normalization & Saving ---
        stereo_noise = np.stack([noise_ch1, noise_ch2], axis=1)
        stereo_mix = np.stack([final_ch1, final_ch2], axis=1)
        stereo_target = np.stack([target_ch1, target_ch2], axis=1)
        stereo_interf = np.stack([interf_ch1_total, interf_ch2_total], axis=1)
        
        peak = np.max(np.abs(stereo_mix)) + 1e-9
        
        stereo_mix /= peak
        stereo_target /= peak
        stereo_interf /= peak
        stereo_noise /= peak

        if save_outputs:
            sf.write("mixture.wav", stereo_mix, self.fs)
            sf.write("target.wav", stereo_target, self.fs)
            
        return stereo_mix, stereo_target, stereo_interf, stereo_noise


# --- Backward Compatibility Aliases and Top-Level Functions ---
FS = AcousticSceneSimulator.DEFAULT_FS
ROOM_DIM = AcousticSceneSimulator.DEFAULT_ROOM_DIM
DEFAULT_RT60 = AcousticSceneSimulator.DEFAULT_RT60
SNR_TARGET_DB = AcousticSceneSimulator.DEFAULT_SNR_TARGET_DB
SIR_TARGET_DB = AcousticSceneSimulator.DEFAULT_SIR_TARGET_DB
MIC_LOCS = AcousticSceneSimulator.DEFAULT_MIC_LOCS

get_audio_files = AudioDatasetFetcher.get_audio_files
get_single_dataset_files = AudioDatasetFetcher.get_single_dataset_files
add_awgn = AcousticSceneSimulator.add_awgn


def simulate_scene(n=1, reverb=True, target_rt60=0.5, interferer_angles=None, save_outputs=False):
    """Functional wrapper for AcousticSceneSimulator.simulate for backward compatibility."""
    simulator = AcousticSceneSimulator()
    return simulator.simulate(
        n=n,
        reverb=reverb,
        target_rt60=target_rt60,
        interferer_angles=interferer_angles,
        save_outputs=save_outputs
    )
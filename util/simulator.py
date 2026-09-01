import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve


class AudioDatasetFetcher:
    """Handles fetching and discovery of audio datasets from Kaggle / local paths."""

    @staticmethod
    def get_single_dataset_files(dataset_name, n_needed, min_duration=5.0):
        """Fetches n_needed files from a specific dataset."""
        files = []
        try:
            if dataset_name == 'librispeech':
                path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
                files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
            elif dataset_name == 'musan_speech':
                path = kagglehub.dataset_download("dogrose/musan-dataset")
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

            if len(valid_files) < n_needed:
                while len(valid_files) < n_needed:
                    valid_files += valid_files
                valid_files = valid_files[:n_needed]
            
            return valid_files
        except Exception as e:
            print(f"Error getting data: {e}")
            return []


class AcousticSceneSimulator:
    """
    Object-oriented simulator for shoebox room acoustics, arbitrary M-channel
    arrays, arbitrary N-source mixtures, reverberation, and calibrated noise.
    """

    DEFAULT_FS = 16000
    DEFAULT_ROOM_DIM = [4.9, 4.9, 4.9]
    DEFAULT_RT60 = 0.5
    DEFAULT_SNR_TARGET_DB = 25
    DEFAULT_SIR_TARGET_DB = 0
    DEFAULT_MIC_SPACING = 0.08  # 8 cm
    DEFAULT_RADIUS = 2.0
    DEFAULT_MIN_SRC_DIST = 0.5

    def __init__(
        self,
        num_mics=2,
        mic_spacing=DEFAULT_MIC_SPACING,
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
        self.num_mics = num_mics
        self.mic_spacing = mic_spacing
        self.room_dim = room_dim if room_dim is not None else list(self.DEFAULT_ROOM_DIM)
        self.default_rt60 = default_rt60
        self.snr_target_db = snr_target_db
        self.sir_target_db = sir_target_db
        self.radius = radius
        self.min_src_dist = min_src_dist
        self.dataset_fetcher = dataset_fetcher if dataset_fetcher is not None else AudioDatasetFetcher()
        self.room = None

        # Build or assign microphone locations (3, M)
        if mic_locs is not None:
            self.mic_locs = np.asarray(mic_locs)
            self.num_mics = self.mic_locs.shape[1]
        else:
            self.mic_locs = self.generate_ula(self.num_mics, self.mic_spacing, self.room_dim)

    @staticmethod
    def generate_ula(num_mics, spacing, room_dim, height=1.5):
        """Generates a Uniform Linear Array centered along the X-axis of the room."""
        center_x = room_dim[0] / 2.0
        center_y = room_dim[1] / 2.0
        
        # Calculate start position along x-axis to keep array perfectly centered
        total_aperture = (num_mics - 1) * spacing
        start_x = center_x - (total_aperture / 2.0)
        
        x_coords = [start_x + i * spacing for i in range(num_mics)]
        y_coords = [center_y] * num_mics
        z_coords = [height] * num_mics
        
        return np.array([x_coords, y_coords, z_coords])

    @staticmethod
    def add_awgn(signal, snr_db, ref_signal=None):
        """Adds sensor AWGN scaled strictly to the target reference signal power."""
        if ref_signal is None:
            ref_signal = signal
            
        sig_power = np.mean(ref_signal ** 2)
        if sig_power == 0:
            return signal, np.zeros(signal.shape)
        
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        return signal + noise, noise

    def load_audio_sources(self, n=1):
        """Loads and zero-pads target (LJSpeech) and N interferers (LibriSpeech/MUSAN)."""
        target_files = self.dataset_fetcher.get_single_dataset_files('ljspeech', 1)
        
        interferer_files = []
        if n > 0:
            interferer_files += self.dataset_fetcher.get_single_dataset_files('librispeech', n)
            # If LibriSpeech ran short, top up with MUSAN
            if len(interferer_files) < n:
                interferer_files += self.dataset_fetcher.get_single_dataset_files('musan_speech', n - len(interferer_files))
            
        files = target_files + interferer_files[:n]

        sigs = []
        max_len = 0
        for f in files:
            y, _ = librosa.load(f, sr=self.fs, mono=True)
            sigs.append(y)
            if len(y) > max_len:
                max_len = len(y)
        
        # Zero-pad to identical duration
        sigs = [np.pad(s, (0, max_len - len(s)), 'constant') for s in sigs]
        target_sig = sigs[0]
        interferer_sigs = sigs[1:]
        return target_sig, interferer_sigs

    def setup_room(self, reverb=True, target_rt60=None):
        """Initializes ShoeBox room and adds the M-element microphone array."""
        rt60 = target_rt60 if target_rt60 is not None else self.default_rt60
        if reverb and rt60 > 0.0:
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
        """Places target (at 90 deg) and N interferers around the array center."""
        mic_center = np.mean(self.mic_locs, axis=1) 
        placed_sources = []

        # Target fixed at broadside (90 degrees)
        target_angle = np.pi / 2 
        pos_target = [
            mic_center[0] + self.radius * np.cos(target_angle),
            mic_center[1] + self.radius * np.sin(target_angle),
            1.5
        ]
        room.add_source(pos_target)
        placed_sources.append(np.array(pos_target))

        # Place N interferers
        if n > 0:
            for i in range(n):
                if interferer_angles is not None and len(interferer_angles) > i:
                    theta = interferer_angles[i] * (np.pi / 180)
                    cand_x = mic_center[0] + self.radius * np.cos(theta)
                    cand_y = mic_center[1] + self.radius * np.sin(theta)
                    candidate = np.array([cand_x, cand_y, 1.5])
                    room.add_source(candidate)
                    placed_sources.append(candidate)
                else:
                    max_attempts = 100
                    for attempt in range(max_attempts):
                        theta = random.uniform(0, 2 * np.pi)
                        cand_x = mic_center[0] + self.radius * np.cos(theta)
                        cand_y = mic_center[1] + self.radius * np.sin(theta)
                        candidate = np.array([cand_x, cand_y, 1.5])
                        
                        collision = any(np.linalg.norm(candidate - es) < self.min_src_dist for es in placed_sources)
                        if not collision:
                            room.add_source(candidate)
                            placed_sources.append(candidate)
                            break
                    
                    if len(placed_sources) <= i + 1:
                        raise RuntimeError(f"Could not place interferer {i+1} without collision.")
                        
        return placed_sources

    def simulate(self, n=1, reverb=True, target_rt60=0.5, interferer_angles=None, save_outputs=False):
        """
        Simulates the entire scene for arbitrary M microphones and N interferers.
        Returns:
            mix, target, interferer, noise as arrays of shape (Samples, M).
        """
        target_sig, interferer_sigs = self.load_audio_sources(n=n)
        room = self.setup_room(reverb=reverb, target_rt60=target_rt60)
        self.place_sources(room, n=n, interferer_angles=interferer_angles)
        room.compute_rir()

        M = self.num_mics

        def get_convolved(sig, rir):
            return fftconvolve(sig, rir, mode='full')
            
        def match_length(sig, target_length):
            if len(sig) > target_length:
                return sig[:target_length]
            elif len(sig) < target_length:
                return np.pad(sig, (0, target_length - len(sig)), 'constant')
            return sig

        # Convolve Target across all M microphones (Source Index 0)
        target_convolved = [get_convolved(target_sig, room.rir[m][0]) for m in range(M)]
        final_len = max(len(tc) for tc in target_convolved)
        
        target_mics = np.zeros((M, final_len))
        for m in range(M):
            target_mics[m] = match_length(target_convolved[m], final_len)

        # Convolve and accumulate all N Interferers across all M microphones
        interf_mics = np.zeros((M, final_len))
        if n > 0:
            for src_idx, i_sig in enumerate(interferer_sigs, start=1):
                for m in range(M):
                    i_conv = get_convolved(i_sig, room.rir[m][src_idx])
                    interf_mics[m] += match_length(i_conv, final_len)

            # SIR calibration calibrated to Reference Mic (Mic 0)
            p_target_ref = np.mean(target_mics[0] ** 2)
            p_interf_ref = np.mean(interf_mics[0] ** 2)
            
            if p_interf_ref > 0:
                scaling_factor = np.sqrt(p_target_ref / (p_interf_ref * (10 ** (self.sir_target_db / 10))))
                interf_mics *= scaling_factor

        # Mix target and interference
        clean_mix = target_mics + interf_mics

        # Add AWGN noise channel-by-channel scaled to that channel's target power
        noise_mics = np.zeros((M, final_len))
        final_mix = np.zeros((M, final_len))
        for m in range(M):
            final_mix[m], noise_mics[m] = self.add_awgn(
                clean_mix[m], self.snr_target_db, ref_signal=target_mics[m]
            )

        # Transpose to (Samples, M)
        stereo_mix = final_mix.T
        stereo_target = target_mics.T
        stereo_interf = interf_mics.T
        stereo_noise = noise_mics.T

        # Peak normalization to [-1, 1] across all channels
        peak = np.max(np.abs(stereo_mix)) + 1e-9
        stereo_mix /= peak
        stereo_target /= peak
        stereo_interf /= peak
        stereo_noise /= peak

        if save_outputs:
            sf.write("mixture.wav", stereo_mix, self.fs)
            sf.write("target.wav", stereo_target, self.fs)
            
        return stereo_mix, stereo_target, stereo_interf, stereo_noise
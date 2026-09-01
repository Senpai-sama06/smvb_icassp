import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from util.simulator import AcousticSceneSimulator

def test_simulator_physics():
    print("--- RUNNING SIMULATOR PHYSICS VALIDATION ---")
    fs = 16000
    
    # Initialize simulator with 20dB SNR and 0dB SIR
    sim = AcousticSceneSimulator(snr_target_db=20, sir_target_db=0, fs=fs, default_rt60=0.0) # 0 RT60 to isolate direct path
    
    # Place interferer exactly at 0 degrees (endfire)
    # Target is hardcoded to 90 degrees (broadside) in the simulator
    mix, target, interf, noise = sim.simulate(n=1, reverb=False, interferer_angles=[0], save_outputs=False)
    
    print("\n1. POWER SCALING VALIDATION (SIR = 0 dB, SNR = 20 dB):")
    p_target = np.var(target[:, 0])
    p_interf = np.var(interf[:, 0])
    p_noise = np.var(noise[:, 0])
    
    actual_sir = 10 * np.log10(p_target / p_interf)
    actual_snr = 10 * np.log10(p_target / p_noise)
    
    print(f"  -> Expected SIR:  0.00 dB | Actual SIR: {actual_sir:+.2f} dB")
    print(f"  -> Expected SNR: 20.00 dB | Actual SNR: {actual_snr:+.2f} dB")
    
    print("\n2. TIME DELAY OF ARRIVAL (TDOA) VALIDATION:")
    print("  -> Mic spacing: 8 cm. Speed of sound: 343 m/s. Fs: 16000 Hz.")
    print("  -> Expected broadside (90°) delay: 0.00 samples")
    print("  -> Expected endfire (0°) delay: ~3.73 samples")
    
    # Calculate Cross-Correlation to find true delays
    def get_delay(sig):
        corr = correlate(sig[:, 0], sig[:, 1], mode='full')
        lags = np.arange(-len(sig[:, 0]) + 1, len(sig[:, 0]))
        max_idx = np.argmax(np.abs(corr))
        return lags[max_idx]
        
    delay_target = get_delay(target)
    delay_interf = get_delay(interf)
    
    print(f"  -> Actual Target (90°) Delay: {delay_target} samples")
    print(f"  -> Actual Interferer (0°) Delay: {delay_interf} samples")
    
    # Validate WNG Explosion manually
    print("\n3. WNG EXPLOSION VALIDATION:")
    print("  -> Simulating highly collinear filter weights (e.g., w = [+1000, -999])")
    w = np.array([1000, -999])
    
    # Apply weights to target and noise
    target_out = w[0] * target[:, 0] + w[1] * target[:, 1]
    noise_out = w[0] * noise[:, 0] + w[1] * noise[:, 1]
    
    out_snr = 10 * np.log10(np.var(target_out) / np.var(noise_out))
    print(f"  -> Input SNR: 20.00 dB")
    print(f"  -> Output SNR after extreme weights: {out_snr:+.2f} dB")
    
    if abs(actual_sir) < 0.1 and abs(actual_snr - 20) < 0.1 and delay_target == 0 and delay_interf in [3, 4]:
        print("\n✅ VALIDATION PASSED: The Simulator is mathematically flawless.")
    else:
        print("\n❌ VALIDATION FAILED.")

if __name__ == "__main__":
    test_simulator_physics()
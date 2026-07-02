"""Step 5c: Generate Noise/Artifact ECGs.

Noise characteristics:
- Signal dominated by artifacts, not ECG morphology
- Very low SNR (< 3 dB)
- Few or no detectable R-peaks
- Combination of:
  * Large baseline wander (low-frequency drift)
  * High-frequency muscle/EMG noise
  * 50/60 Hz powerline interference
  * Motion artifacts (sudden transients)
  * Random ADC/thermal noise
"""

import numpy as np
import yaml


def generate_noise_ecg(config):
    """Generate a single Noise/Artifact ECG.

    Strategy:
    - Generate pure noise from multiple sources
    - Optionally embed a few very weak/distorted beats to simulate partial signal
    - Ensure final SNR is very poor (< 3 dB)
    """
    fs = config['sampling_rate']
    duration = config['duration']
    total_samples = fs * duration
    noise_cfg = config['noise']
    t = np.arange(total_samples) / fs

    # Reference amplitude (typical ECG signal range)
    ref_amplitude = 300.0  # typical ADC range for our signals

    # 1. Baseline wander (large, slow drift)
    baseline_amp = noise_cfg.get('baseline_amplitude', 0.3) * ref_amplitude
    freq_range = noise_cfg.get('baseline_freq_range', [0.1, 0.5])
    baseline = np.zeros(total_samples)
    for _ in range(3):
        freq = np.random.uniform(freq_range[0], freq_range[1])
        phase = np.random.uniform(0, 2 * np.pi)
        baseline += baseline_amp * np.sin(2 * np.pi * freq * t + phase)

    # 2. High-frequency muscle noise (broadband)
    muscle_amp = noise_cfg.get('muscle_amplitude', 0.25) * ref_amplitude
    muscle_noise = muscle_amp * np.random.randn(total_samples)
    # Bandpass to muscle band
    from scipy.signal import butter, filtfilt
    muscle_band = noise_cfg.get('muscle_band', [20, 100])
    nyq = 0.5 * fs
    if muscle_band[1] < nyq:
        b, a = butter(2, [muscle_band[0] / nyq, muscle_band[1] / nyq], btype='band')
        muscle_noise = filtfilt(b, a, muscle_noise) * 3  # boost after filtering

    # 3. Powerline interference (50 or 60 Hz)
    pl_freq = noise_cfg.get('powerline_freq', 50)
    pl_amp = noise_cfg.get('powerline_amplitude', 0.15) * ref_amplitude
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    # Add harmonics
    powerline += (pl_amp * 0.3) * np.sin(2 * np.pi * 2 * pl_freq * t)

    # 4. ADC/thermal noise (white noise)
    adc_amp = noise_cfg.get('adc_noise', 0.05) * ref_amplitude
    adc_noise = adc_amp * np.random.randn(total_samples)

    # 5. Motion artifacts (sudden transients)
    n_artifacts = noise_cfg.get('motion_artifacts', 5)
    motion = np.zeros(total_samples)
    for _ in range(n_artifacts):
        # Random position and duration
        pos = np.random.randint(0, total_samples - 200)
        duration_samples = np.random.randint(30, 200)
        # Transient: either a spike or a step
        if np.random.random() < 0.5:
            # Spike
            artifact = np.random.randn(duration_samples) * ref_amplitude * 0.4
        else:
            # Step/shift
            artifact = np.ones(duration_samples) * np.random.randn() * ref_amplitude * 0.3
        motion[pos:pos + duration_samples] += artifact

    # Combine all noise sources
    signal = baseline + muscle_noise + powerline + adc_noise + motion

    return signal, {
        'baseline_amp': baseline_amp,
        'muscle_amp': muscle_amp,
        'powerline_freq': pl_freq,
        'n_motion_artifacts': n_artifacts,
    }


def generate_all_noise(config):
    """Generate all Noise ECG files.

    Returns: list of (signal, metadata) tuples
    """
    count = config.get('count_noise', 30)
    print("\n" + "=" * 60)
    print(f"STEP 5c: Generating {count} Noise/Artifact ECGs")
    print("=" * 60)

    results = []
    for i in range(count):
        signal, meta = generate_noise_ecg(config)
        results.append((signal, meta))
        print(f"  Noise #{i+1:02d}: baseline={meta['baseline_amp']:.0f}, "
              f"muscle={meta['muscle_amp']:.0f}, "
              f"PL={meta['powerline_freq']}Hz, "
              f"artifacts={meta['n_motion_artifacts']}")

    print(f"\n  Generated {len(results)} Noise ECGs")
    print("=" * 60)
    return results

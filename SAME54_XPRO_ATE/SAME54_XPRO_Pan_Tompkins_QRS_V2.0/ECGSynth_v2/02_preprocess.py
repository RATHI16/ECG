"""Step 2: Preprocess ECG signals.

- Apply bandpass filter (0.5-45 Hz) to remove baseline drift and high-freq noise
- Normalize signals
- Resample to target sampling rate if needed
"""

import numpy as np
import yaml
from ecg_utils import bandpass_filter


def preprocess_signal(signal, fs, config):
    """Clean a raw ECG signal.

    Returns: cleaned signal at target sampling rate.
    """
    filter_cfg = config.get('filter', {})
    lowcut = filter_cfg.get('lowcut', 0.5)
    highcut = filter_cfg.get('highcut', 45.0)
    order = filter_cfg.get('order', 3)
    target_fs = config['sampling_rate']

    # Apply bandpass filter
    filtered = bandpass_filter(signal, fs, lowcut=lowcut, highcut=highcut, order=order)

    # Resample if necessary
    if abs(fs - target_fs) > 5:
        n_samples_new = int(len(filtered) * target_fs / fs)
        from scipy.signal import resample
        filtered = resample(filtered, n_samples_new)

    return filtered


def preprocess_all(valid_recordings, config):
    """Preprocess all validated recordings.

    Args:
        valid_recordings: list of (filepath, signal, fs) from validation step
        config: loaded config dict

    Returns: list of (filepath, cleaned_signal, fs) tuples.
    """
    target_fs = config['sampling_rate']

    preprocessed = []
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing ECG Signals")
    print("=" * 60)

    for filepath, signal, fs in valid_recordings:
        cleaned = preprocess_signal(signal, fs, config)
        fname = filepath.split('/')[-1].split('\\')[-1]
        print(f"  Processed: {fname} -> {len(cleaned)} samples at {target_fs} Hz")
        preprocessed.append((filepath, cleaned, target_fs))

    print(f"\n  Preprocessed {len(preprocessed)} recordings")
    print("=" * 60)
    return preprocessed


if __name__ == "__main__":
    from step01_validate import validate_all_recordings
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    valid = validate_all_recordings(cfg)
    preprocessed = preprocess_all(valid, cfg)

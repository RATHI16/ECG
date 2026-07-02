"""Step 1: Validate raw ECG recordings.

Checks:
- File exists and is readable
- Has correct CSV format (timestamp, New ECG Sample)
- No NaN values
- Signal amplitude in reasonable range
- Minimum length (at least 2 seconds of data)
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml


def load_ecg_file(filepath):
    """Load ECG CSV file and return signal array and metadata."""
    df = pd.read_csv(filepath)

    # Handle column naming variations
    if 'New ECG Sample' in df.columns:
        signal_col = 'New ECG Sample'
    elif 'amplitude' in df.columns:
        signal_col = 'amplitude'
    else:
        signal_col = df.columns[1]  # assume second column is signal

    signal = df[signal_col].values.astype(float)
    timestamps = df[df.columns[0]].values.astype(float)

    # Estimate sampling rate from timestamps
    dt = np.median(np.diff(timestamps))
    estimated_fs = 1.0 / dt if dt > 0 else 1000

    return signal, timestamps, estimated_fs


def validate_recording(filepath, config):
    """Validate a single ECG recording.

    Returns: (is_valid, signal, estimated_fs, issues_list)
    """
    issues = []
    expected_fs = config['sampling_rate']
    min_duration = 2.0  # minimum seconds

    if not os.path.exists(filepath):
        return False, None, None, [f"File not found: {filepath}"]

    try:
        signal, timestamps, estimated_fs = load_ecg_file(filepath)
    except Exception as e:
        return False, None, None, [f"Failed to read: {e}"]

    # Check for NaN values
    nan_count = np.isnan(signal).sum()
    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN values")
        signal = signal[~np.isnan(signal)]

    # Check minimum length
    duration = len(signal) / estimated_fs
    if duration < min_duration:
        issues.append(f"Too short: {duration:.1f}s (need >= {min_duration}s)")
        return False, signal, estimated_fs, issues

    # Check sampling rate
    fs_tolerance = 0.1  # 10% tolerance
    if abs(estimated_fs - expected_fs) / expected_fs > fs_tolerance:
        issues.append(f"Sampling rate {estimated_fs:.0f} Hz differs from expected {expected_fs} Hz")

    # Check amplitude range
    sig_min, sig_max = signal.min(), signal.max()
    if sig_min < -5000 or sig_max > 5000:
        issues.append(f"Extreme amplitude range: [{sig_min:.0f}, {sig_max:.0f}]")

    # Check for constant signal (dead channel)
    if np.std(signal) < 1.0:
        issues.append("Signal appears flat/constant")
        return False, signal, estimated_fs, issues

    is_valid = len([i for i in issues if "Too short" in i or "flat" in i]) == 0
    return is_valid, signal, estimated_fs, issues


def validate_all_recordings(config):
    """Validate all input recordings. Returns list of valid (filepath, signal, fs) tuples."""
    input_dir = config.get('input_dir', '../')
    input_files = config.get('input_files', [])

    valid_recordings = []
    print("=" * 60)
    print("STEP 1: Validating Raw ECG Recordings")
    print("=" * 60)

    for fname in input_files:
        filepath = os.path.join(input_dir, fname)
        is_valid, signal, fs, issues = validate_recording(filepath, config)

        status = "PASS" if is_valid else "FAIL"
        print(f"\n  [{status}] {fname}")
        if signal is not None:
            duration = len(signal) / fs if fs else 0
            print(f"       Samples: {len(signal)}, Duration: {duration:.1f}s, Fs: {fs:.0f} Hz")
            print(f"       Range: [{signal.min():.0f}, {signal.max():.0f}], Std: {np.std(signal):.1f}")
        if issues:
            for issue in issues:
                print(f"       WARNING: {issue}")

        if is_valid:
            valid_recordings.append((filepath, signal, fs))

    print(f"\n  Result: {len(valid_recordings)}/{len(input_files)} recordings valid")
    print("=" * 60)
    return valid_recordings


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    valid = validate_all_recordings(cfg)
    print(f"\nValidated {len(valid)} recordings ready for processing.")

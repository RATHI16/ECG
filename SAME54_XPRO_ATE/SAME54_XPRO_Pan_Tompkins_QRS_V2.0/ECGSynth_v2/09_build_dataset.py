"""Step 6b: Build final dataset - save CSVs, metadata, and reports."""

import os
import numpy as np
import pandas as pd
import yaml


def save_ecg_csv(signal, filepath, fs):
    """Save an ECG signal to CSV in the same format as source files.

    Format: timestamp,New ECG Sample
    Timestamps start at 0.0, incrementing by 1/fs.
    Signal values are integers (matching ADC output format).
    """
    n_samples = len(signal)
    timestamps = np.arange(n_samples) / fs
    # Convert to integer ADC values
    signal_int = np.round(signal).astype(int)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'New ECG Sample': signal_int,
    })
    df.to_csv(filepath, index=False)


def build_dataset(valid_normal, valid_afib, valid_noise, config, output_dir="."):
    """Save all validated ECGs to CSV files and create metadata.

    Directory structure:
        output_dir/
            Normal/normal_001.csv ... normal_030.csv
            AFib/afib_001.csv ... afib_030.csv
            Noise/noise_001.csv ... noise_030.csv
            metadata.csv
    """
    fs = config['sampling_rate']

    print("\n" + "=" * 60)
    print("STEP 6b: Building Final Dataset")
    print("=" * 60)

    # Create directories
    for subdir in ['Normal', 'AFib', 'Noise']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    metadata_rows = []

    # Save Normal ECGs
    print(f"\n  Saving {len(valid_normal)} Normal ECGs...")
    for i, (signal, meta, metrics) in enumerate(valid_normal[:30]):
        fname = f"normal_{i+1:03d}.csv"
        fpath = os.path.join(output_dir, 'Normal', fname)
        save_ecg_csv(signal, fpath, fs)
        metadata_rows.append({
            'filename': f"Normal/{fname}",
            'class': 'Normal',
            'hr_bpm': metrics.get('hr_bpm', meta.get('hr_bpm', np.nan)),
            'rr_cv': metrics.get('rr_cv', 0),
            'snr_db': metrics.get('snr_db', np.nan),
            'sdnn_ms': metrics.get('sdnn', np.nan),
            'n_samples': len(signal),
            'duration_s': len(signal) / fs,
        })

    # Save AFib ECGs
    print(f"  Saving {len(valid_afib)} AFib ECGs...")
    for i, (signal, meta, metrics) in enumerate(valid_afib[:30]):
        fname = f"afib_{i+1:03d}.csv"
        fpath = os.path.join(output_dir, 'AFib', fname)
        save_ecg_csv(signal, fpath, fs)
        metadata_rows.append({
            'filename': f"AFib/{fname}",
            'class': 'AFib',
            'hr_bpm': metrics.get('hr_bpm', meta.get('hr_bpm', np.nan)),
            'rr_cv': metrics.get('rr_cv', 0),
            'snr_db': metrics.get('snr_db', np.nan),
            'sdnn_ms': metrics.get('sdnn', np.nan),
            'qrs_width_ms': metrics.get('qrs_width_ms', np.nan),
            'n_samples': len(signal),
            'duration_s': len(signal) / fs,
        })

    # Save Noise ECGs
    print(f"  Saving {len(valid_noise)} Noise ECGs...")
    for i, (signal, meta, metrics) in enumerate(valid_noise[:30]):
        fname = f"noise_{i+1:03d}.csv"
        fpath = os.path.join(output_dir, 'Noise', fname)
        save_ecg_csv(signal, fpath, fs)
        metadata_rows.append({
            'filename': f"Noise/{fname}",
            'class': 'Noise',
            'hr_bpm': np.nan,  # not meaningful for noise
            'rr_cv': np.nan,
            'snr_db': metrics.get('snr_db', np.nan),
            'n_peaks_detected': metrics.get('n_peaks', 0),
            'n_samples': len(signal),
            'duration_s': len(signal) / fs,
        })

    # Save metadata
    metadata = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(output_dir, 'metadata.csv')
    metadata.to_csv(meta_path, index=False)

    total = len(valid_normal[:30]) + len(valid_afib[:30]) + len(valid_noise[:30])
    print(f"\n  Dataset saved: {total} ECG files")
    print(f"  Normal: {len(valid_normal[:30])}/30")
    print(f"  AFib:   {len(valid_afib[:30])}/30")
    print(f"  Noise:  {len(valid_noise[:30])}/30")
    print(f"  Metadata: {meta_path}")
    print("=" * 60)

    return metadata

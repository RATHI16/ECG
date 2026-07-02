"""Step 3: Extract per-beat features from preprocessed ECGs.

For each recording:
- Detect R-peaks
- Segment individual beats (templates)
- Compute beat-level features: HR, RR, QRS width, amplitudes
- Compute HRV metrics: SDNN, RMSSD, pNN50, CV
- Save beat templates and parameter database
"""

import numpy as np
import pandas as pd
import yaml
import neurokit2 as nk
from ecg_utils import detect_rpeaks, segment_beats, compute_rr_intervals, compute_hrv_metrics


def extract_features_from_signal(signal, fs):
    """Extract beat-level features from a single ECG signal.

    Returns:
        beats: list of beat template arrays
        features: dict with arrays of per-beat measurements
        hrv: dict with HRV metrics
    """
    # Detect R-peaks
    rpeaks = detect_rpeaks(signal, fs)

    if len(rpeaks) < 3:
        return [], {}, {}

    # Segment beats
    beats = segment_beats(signal, rpeaks, fs)

    # RR intervals
    rr_ms = compute_rr_intervals(rpeaks, fs)

    # HRV metrics
    hrv = compute_hrv_metrics(rr_ms)

    # Per-beat features
    hr_values = 60000.0 / rr_ms  # instantaneous HR
    qrs_widths = []
    r_amplitudes = []
    beat_lengths = []

    for i, rpeak in enumerate(rpeaks):
        # R amplitude
        r_amplitudes.append(signal[rpeak])

        # Estimate QRS width from the peak region
        # Look for zero-crossings around R-peak
        window = int(0.08 * fs)  # 80ms window each side
        start = max(0, rpeak - window)
        end = min(len(signal), rpeak + window)
        segment = signal[start:end]

        # QRS width: approximate as region above 50% of R-peak amplitude
        r_amp = signal[rpeak]
        threshold = r_amp * 0.5 if r_amp > 0 else r_amp * 1.5
        if r_amp > 0:
            above_thresh = segment > threshold
        else:
            above_thresh = segment < threshold
        qrs_width_samples = np.sum(above_thresh)
        qrs_widths.append(qrs_width_samples / fs * 1000)  # convert to ms

    for beat in beats:
        beat_lengths.append(len(beat) / fs * 1000)  # ms

    features = {
        'rr_intervals_ms': rr_ms,
        'hr_bpm': hr_values,
        'qrs_width_ms': np.array(qrs_widths[:len(rr_ms)]),
        'r_amplitude': np.array(r_amplitudes[:len(rr_ms)]),
        'beat_length_ms': np.array(beat_lengths),
    }

    return beats, features, hrv


def extract_all_features(preprocessed_recordings, config):
    """Extract features from all preprocessed recordings.

    Args:
        preprocessed_recordings: list of (filepath, signal, fs) tuples

    Returns:
        all_beats: list of all beat templates
        parameter_db: DataFrame with all beat parameters
        summary: dict with aggregate statistics
    """
    fs = config['sampling_rate']
    all_beats = []
    all_features = []

    print("\n" + "=" * 60)
    print("STEP 3: Extracting Beat Features")
    print("=" * 60)

    for filepath, signal, _ in preprocessed_recordings:
        fname = filepath.split('/')[-1].split('\\')[-1]
        beats, features, hrv = extract_features_from_signal(signal, fs)

        if not beats:
            print(f"  SKIP: {fname} - insufficient R-peaks detected")
            continue

        all_beats.extend(beats)
        n_beats = len(beats)
        mean_hr = np.mean(features['hr_bpm']) if len(features['hr_bpm']) > 0 else 0

        print(f"  {fname}: {n_beats} beats, HR={mean_hr:.0f} bpm, "
              f"SDNN={hrv.get('sdnn', 0):.1f}ms, CV={hrv.get('cv', 0):.3f}")

        # Build per-beat feature rows
        n_rows = min(len(features['rr_intervals_ms']), len(beats))
        for i in range(n_rows):
            row = {
                'source_file': fname,
                'beat_index': i,
                'rr_ms': features['rr_intervals_ms'][i] if i < len(features['rr_intervals_ms']) else np.nan,
                'hr_bpm': features['hr_bpm'][i] if i < len(features['hr_bpm']) else np.nan,
                'qrs_width_ms': features['qrs_width_ms'][i] if i < len(features['qrs_width_ms']) else np.nan,
                'r_amplitude': features['r_amplitude'][i] if i < len(features['r_amplitude']) else np.nan,
                'beat_length_ms': features['beat_length_ms'][i] if i < len(features['beat_length_ms']) else np.nan,
                'sdnn_ms': hrv.get('sdnn', np.nan),
                'rmssd_ms': hrv.get('rmssd', np.nan),
                'pnn50': hrv.get('pnn50', np.nan),
                'cv': hrv.get('cv', np.nan),
            }
            all_features.append(row)

    parameter_db = pd.DataFrame(all_features)

    # Summary statistics
    summary = {
        'total_beats': len(all_beats),
        'total_recordings': len(preprocessed_recordings),
        'mean_hr': parameter_db['hr_bpm'].mean(),
        'std_hr': parameter_db['hr_bpm'].std(),
        'mean_rr': parameter_db['rr_ms'].mean(),
        'mean_qrs': parameter_db['qrs_width_ms'].mean(),
    }

    print(f"\n  Total beats extracted: {len(all_beats)}")
    print(f"  Mean HR: {summary['mean_hr']:.1f} +/- {summary['std_hr']:.1f} bpm")
    print(f"  Mean RR: {summary['mean_rr']:.1f} ms")
    print(f"  Mean QRS width: {summary['mean_qrs']:.1f} ms")
    print("=" * 60)

    return all_beats, parameter_db, summary


if __name__ == "__main__":
    from ecg_utils import bandpass_filter
    from signal_quality import compute_snr_simple
    import sys
    sys.path.insert(0, '.')
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    # Would need preprocessed data from step 2
    print("Run via run_pipeline.py for full pipeline execution.")

"""Diagnose which Normal/AFib files look like noise and why."""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import welch
from scipy.stats import kurtosis

FS = 1000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def analyze_file(filepath, fs=1000):
    """Analyze a single file for noise-like characteristics."""
    df = pd.read_csv(filepath)
    signal = df.iloc[:, 1].values.astype(float)

    # Basic stats
    kurt = kurtosis(signal, fisher=True)
    ptp = np.ptp(signal)
    std = np.std(signal)

    # Try R-peak detection
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        n_peaks = len(rpeaks)
    except:
        rpeaks = []
        n_peaks = 0

    # Check if QRS peaks are clearly visible
    if n_peaks >= 2:
        r_amps = [abs(signal[r]) for r in rpeaks]
        mean_r_amp = np.mean(r_amps)
        # Compare R-peak amplitude to baseline noise
        baseline_std = np.std(signal[signal < np.percentile(signal, 75)])
        snr_qrs = mean_r_amp / baseline_std if baseline_std > 0 else 0
    else:
        mean_r_amp = 0
        snr_qrs = 0

    # Frequency analysis
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    from numpy import trapezoid
    total = trapezoid(psd, freqs)
    ecg_mask = (freqs >= 0.5) & (freqs <= 40)
    ecg_power = trapezoid(psd[ecg_mask], freqs[ecg_mask])
    hf_mask = (freqs >= 40) & (freqs <= 200)
    hf_power = trapezoid(psd[hf_mask], freqs[hf_mask])
    ecg_ratio = ecg_power / total if total > 0 else 0
    hf_ratio = hf_power / total if total > 0 else 0

    # Visual quality score: does it look like ECG or noise?
    # Good ECG: high kurtosis (>5), clear R-peaks (snr_qrs > 3), ECG band dominant
    looks_like_ecg = (kurt > 3 and n_peaks >= 5 and snr_qrs > 2 and ecg_ratio > 0.7)
    looks_like_noise = (kurt < 2 or n_peaks < 3 or snr_qrs < 1.5 or hf_ratio > 0.3)

    return {
        'kurtosis': kurt,
        'n_peaks': n_peaks,
        'snr_qrs': snr_qrs,
        'mean_r_amp': mean_r_amp,
        'ecg_ratio': ecg_ratio,
        'hf_ratio': hf_ratio,
        'ptp': ptp,
        'std': std,
        'looks_like_ecg': looks_like_ecg,
        'looks_like_noise': looks_like_noise,
    }


def main():
    print("=" * 80)
    print("  DIAGNOSING NOISE-LIKE FILES IN NORMAL AND AFIB CLASSES")
    print("=" * 80)

    # Check Normal files
    print("\n  NORMAL CLASS - Files that look like noise:")
    print(f"  {'File':<20} {'Kurt':>6} {'Peaks':>6} {'SNR_QRS':>8} {'ECG%':>6} {'HF%':>6} {'Verdict':<12}")
    print(f"  {'-'*70}")

    normal_problems = []
    for fname in sorted(os.listdir(os.path.join(BASE_DIR, 'Normal'))):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(BASE_DIR, 'Normal', fname)
        info = analyze_file(fpath)
        verdict = "OK" if info['looks_like_ecg'] else "NOISE-LIKE"
        if not info['looks_like_ecg']:
            normal_problems.append(fname)
        if not info['looks_like_ecg'] or info['kurtosis'] < 5:
            print(f"  {fname:<20} {info['kurtosis']:>6.1f} {info['n_peaks']:>6} "
                  f"{info['snr_qrs']:>8.1f} {info['ecg_ratio']*100:>5.1f}% "
                  f"{info['hf_ratio']*100:>5.1f}% {verdict:<12}")

    print(f"\n  Normal: {len(normal_problems)}/30 look noise-like")

    # Check AFib files
    print("\n\n  AFIB CLASS - Files that look like noise:")
    print(f"  {'File':<20} {'Kurt':>6} {'Peaks':>6} {'SNR_QRS':>8} {'ECG%':>6} {'HF%':>6} {'Verdict':<12}")
    print(f"  {'-'*70}")

    afib_problems = []
    for fname in sorted(os.listdir(os.path.join(BASE_DIR, 'AFib'))):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(BASE_DIR, 'AFib', fname)
        info = analyze_file(fpath)
        verdict = "OK" if info['looks_like_ecg'] else "NOISE-LIKE"
        if not info['looks_like_ecg']:
            afib_problems.append(fname)
        if not info['looks_like_ecg'] or info['kurtosis'] < 5:
            print(f"  {fname:<20} {info['kurtosis']:>6.1f} {info['n_peaks']:>6} "
                  f"{info['snr_qrs']:>8.1f} {info['ecg_ratio']*100:>5.1f}% "
                  f"{info['hf_ratio']*100:>5.1f}% {verdict:<12}")

    print(f"\n  AFib: {len(afib_problems)}/30 look noise-like")

    # Summary
    print("\n\n  ROOT CAUSE ANALYSIS:")
    print("  " + "-" * 60)

    # Check a problematic normal file in detail
    if normal_problems:
        fpath = os.path.join(BASE_DIR, 'Normal', normal_problems[0])
        df = pd.read_csv(fpath)
        sig = df.iloc[:, 1].values.astype(float)
        print(f"\n  Example problem file: {normal_problems[0]}")
        print(f"    Signal range: [{sig.min():.0f}, {sig.max():.0f}]")
        print(f"    First 20 samples: {sig[:20].astype(int).tolist()}")
        # Check if beat templates are producing clean morphology
        # The issue is likely that templates from different recordings
        # are stitched together creating discontinuities

    print("\n\n  ISSUES IDENTIFIED:")
    print("  1. Beat templates from source files may have noisy segments")
    print("  2. Template stitching creates discontinuities at beat boundaries")
    print("  3. f-wave addition in AFib may overwhelm the QRS if amplitude too high")
    print("  4. Need to ensure R-peak amplitude is >> noise floor")


if __name__ == "__main__":
    main()

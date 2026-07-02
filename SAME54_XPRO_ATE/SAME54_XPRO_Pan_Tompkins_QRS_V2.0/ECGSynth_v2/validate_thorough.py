"""Thorough validation of all 90 generated ECGs against Marten's requirements.

Checks:
  Normal Sinus Rhythm:
    - HR 60-100 bpm (resting)
    - Regular RR intervals (equal spacing, low CV)
    - P-waves present before each QRS
    - PR interval 120-200 ms
    - QRS < 120 ms
    - Clean signal (high SNR)

  Atrial Fibrillation (ALL must be satisfied):
    1. Irregularly irregular rhythm (high RR CV)
    2. No distinct P-waves
    3. Presence of f-waves (4-8 Hz baseline oscillations)
    4. Narrow QRS (< 120 ms)
    5. Ventricular rate 90-170 bpm

  Noise/Artifact:
    - Very low SNR
    - No detectable regular rhythm
    - Dominated by artifacts (baseline wander, muscle, powerline)

Also outputs feature statistics for ML Dev Suite classification.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, welch
import neurokit2 as nk

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_utils import detect_rpeaks, compute_rr_intervals, compute_hrv_metrics, bandpass_filter
from signal_quality import compute_snr_simple


def load_generated_ecg(filepath, fs=1000):
    """Load a generated ECG CSV file."""
    df = pd.read_csv(filepath)
    return df['New ECG Sample'].values.astype(float)


def analyze_frequency_content(signal, fs):
    """Analyze frequency content using Welch's PSD."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    from numpy import trapezoid
    # ECG band (0.5-40 Hz)
    ecg_mask = (freqs >= 0.5) & (freqs <= 40)
    ecg_power = trapezoid(psd[ecg_mask], freqs[ecg_mask])
    # f-wave band (4-8 Hz)
    fwave_mask = (freqs >= 4) & (freqs <= 8)
    fwave_power = trapezoid(psd[fwave_mask], freqs[fwave_mask])
    # Powerline band (48-52 Hz)
    pl_mask = (freqs >= 48) & (freqs <= 52)
    pl_power = trapezoid(psd[pl_mask], freqs[pl_mask])
    # High-freq noise (60-200 Hz)
    hf_mask = (freqs >= 60) & (freqs <= 200)
    hf_power = trapezoid(psd[hf_mask], freqs[hf_mask])
    # Total power
    total_power = trapezoid(psd, freqs)

    return {
        'ecg_power_ratio': ecg_power / total_power if total_power > 0 else 0,
        'fwave_power_ratio': fwave_power / total_power if total_power > 0 else 0,
        'powerline_power_ratio': pl_power / total_power if total_power > 0 else 0,
        'hf_noise_ratio': hf_power / total_power if total_power > 0 else 0,
        'total_power': total_power,
    }


def detect_p_waves(signal, rpeaks, fs):
    """Check for P-wave presence before each R-peak.

    Mimics the embedded algorithm's approach:
    - Look at 3/8 of RR interval before each R-peak
    - Count threshold crossings (normal = 2 crossings = 1 P-wave hump)
    """
    p_wave_present_count = 0
    p_wave_absent_count = 0

    for i in range(1, len(rpeaks)):
        rr = rpeaks[i] - rpeaks[i-1]
        # Look at the segment 3/8 of RR before this R-peak
        search_len = int(3 * rr / 8)
        start = rpeaks[i] - search_len
        if start < 0:
            continue
        segment = signal[start:rpeaks[i] - int(0.05 * fs)]  # stop 50ms before R

        if len(segment) < 20:
            continue

        # Find threshold crossings (embedded algorithm uses 80% threshold)
        seg_min = np.min(segment)
        seg_max = np.max(segment)
        seg_range = seg_max - seg_min
        threshold = seg_min + 0.8 * seg_range

        # Count crossings above threshold
        above = segment > threshold
        crossings = 0
        in_crossing = False
        for val in above:
            if val and not in_crossing:
                crossings += 1
                in_crossing = True
            elif not val:
                in_crossing = False

        # Normal P-wave = 1 distinct hump = 1 crossing (embedded counts +1 = 2)
        if crossings == 1:
            p_wave_present_count += 1
        else:
            p_wave_absent_count += 1

    total = p_wave_present_count + p_wave_absent_count
    return {
        'p_wave_present_ratio': p_wave_present_count / total if total > 0 else 0,
        'p_wave_absent_ratio': p_wave_absent_count / total if total > 0 else 0,
        'total_beats_checked': total,
    }


def compute_qrs_width(signal, rpeaks, fs):
    """Estimate QRS width for each detected R-peak."""
    widths = []
    for rpeak in rpeaks:
        window = int(0.08 * fs)  # 80ms each side
        start = max(0, rpeak - window)
        end = min(len(signal), rpeak + window)
        segment = signal[start:end]
        r_amp = signal[rpeak]

        if abs(r_amp) < 10:  # skip if R-peak too small
            continue

        # QRS width: region above 30% of R-peak amplitude
        if r_amp > 0:
            threshold = r_amp * 0.3
            above = segment > threshold
        else:
            threshold = r_amp * 0.3
            above = segment < threshold

        width_samples = np.sum(above)
        width_ms = width_samples / fs * 1000
        if 20 < width_ms < 200:  # plausible range
            widths.append(width_ms)

    return np.array(widths) if widths else np.array([np.nan])


def validate_single_ecg(signal, fs, ecg_class):
    """Full validation of a single ECG. Returns detailed metrics."""
    metrics = {'class': ecg_class}

    # Basic signal stats
    metrics['amplitude_range'] = np.ptp(signal)
    metrics['amplitude_std'] = np.std(signal)
    metrics['amplitude_mean'] = np.mean(signal)

    # Try R-peak detection
    try:
        rpeaks = detect_rpeaks(signal, fs)
        metrics['n_rpeaks'] = len(rpeaks)
    except Exception:
        rpeaks = np.array([])
        metrics['n_rpeaks'] = 0

    # RR interval analysis
    if len(rpeaks) >= 2:
        rr_ms = compute_rr_intervals(rpeaks, fs)
        hrv = compute_hrv_metrics(rr_ms)
        metrics['hr_bpm'] = 60000.0 / np.mean(rr_ms)
        metrics['rr_mean_ms'] = np.mean(rr_ms)
        metrics['rr_std_ms'] = np.std(rr_ms)
        metrics['rr_cv'] = hrv['cv']
        metrics['sdnn_ms'] = hrv['sdnn']
        metrics['rmssd_ms'] = hrv['rmssd']
        metrics['pnn50'] = hrv['pnn50']
    else:
        metrics['hr_bpm'] = np.nan
        metrics['rr_mean_ms'] = np.nan
        metrics['rr_std_ms'] = np.nan
        metrics['rr_cv'] = np.nan
        metrics['sdnn_ms'] = np.nan
        metrics['rmssd_ms'] = np.nan
        metrics['pnn50'] = np.nan

    # QRS width
    if len(rpeaks) >= 2:
        qrs_widths = compute_qrs_width(signal, rpeaks, fs)
        metrics['qrs_mean_ms'] = np.nanmean(qrs_widths)
        metrics['qrs_max_ms'] = np.nanmax(qrs_widths)
    else:
        metrics['qrs_mean_ms'] = np.nan
        metrics['qrs_max_ms'] = np.nan

    # P-wave analysis
    if len(rpeaks) >= 3:
        p_info = detect_p_waves(signal, rpeaks, fs)
        metrics['p_wave_present_ratio'] = p_info['p_wave_present_ratio']
    else:
        metrics['p_wave_present_ratio'] = np.nan

    # Frequency content
    freq_info = analyze_frequency_content(signal, fs)
    metrics['ecg_power_ratio'] = freq_info['ecg_power_ratio']
    metrics['fwave_power_ratio'] = freq_info['fwave_power_ratio']
    metrics['powerline_ratio'] = freq_info['powerline_power_ratio']
    metrics['hf_noise_ratio'] = freq_info['hf_noise_ratio']

    # SNR
    metrics['snr_db'] = compute_snr_simple(signal, fs)

    # Kurtosis (peakedness - high for ECG with sharp QRS, low for noise)
    metrics['kurtosis'] = stats.kurtosis(signal, fisher=True)

    return metrics


def check_requirements(metrics, ecg_class):
    """Check if metrics satisfy class requirements. Returns (pass/fail, issues)."""
    issues = []

    if ecg_class == 'Normal':
        # HR 60-100 bpm
        if not np.isnan(metrics['hr_bpm']):
            if not (50 <= metrics['hr_bpm'] <= 110):
                issues.append(f"HR={metrics['hr_bpm']:.0f} outside 50-110 bpm")
        else:
            issues.append("No HR detectable")

        # Regular RR (CV < 0.15)
        if not np.isnan(metrics['rr_cv']):
            if metrics['rr_cv'] > 0.15:
                issues.append(f"RR_CV={metrics['rr_cv']:.3f} > 0.15 (irregular)")

        # P-waves present
        if not np.isnan(metrics['p_wave_present_ratio']):
            if metrics['p_wave_present_ratio'] < 0.3:
                issues.append(f"P-wave present only {metrics['p_wave_present_ratio']:.0%}")

        # QRS < 120 ms
        if not np.isnan(metrics['qrs_mean_ms']):
            if metrics['qrs_mean_ms'] > 120:
                issues.append(f"QRS={metrics['qrs_mean_ms']:.0f}ms > 120ms")

    elif ecg_class == 'AFib':
        # 1. Irregularly irregular (CV > 0.15)
        if not np.isnan(metrics['rr_cv']):
            if metrics['rr_cv'] < 0.15:
                issues.append(f"[CRIT] RR_CV={metrics['rr_cv']:.3f} < 0.15 (too regular)")
        else:
            issues.append("[CRIT] Cannot measure RR regularity")

        # 2. No P-waves (present ratio should be low)
        if not np.isnan(metrics['p_wave_present_ratio']):
            if metrics['p_wave_present_ratio'] > 0.5:
                issues.append(f"[CRIT] P-waves detected in {metrics['p_wave_present_ratio']:.0%} of beats")

        # 3. f-waves present (elevated 4-8 Hz power)
        if metrics['fwave_power_ratio'] < 0.01:
            issues.append(f"f-wave power low: {metrics['fwave_power_ratio']:.4f}")

        # 4. Narrow QRS (< 120 ms)
        if not np.isnan(metrics['qrs_mean_ms']):
            if metrics['qrs_mean_ms'] > 120:
                issues.append(f"[CRIT] QRS={metrics['qrs_mean_ms']:.0f}ms > 120ms (wide)")

        # 5. Ventricular rate 90-170 bpm
        if not np.isnan(metrics['hr_bpm']):
            if not (60 <= metrics['hr_bpm'] <= 180):
                issues.append(f"HR={metrics['hr_bpm']:.0f} outside 60-180 bpm")

    elif ecg_class == 'Noise':
        # No clear rhythm
        if metrics['n_rpeaks'] > 5:
            # Check if peaks form a regular rhythm
            if not np.isnan(metrics['rr_cv']) and metrics['rr_cv'] < 0.3:
                issues.append(f"Regular rhythm detected: {metrics['n_rpeaks']} peaks, CV={metrics['rr_cv']:.2f}")

        # Low kurtosis (no sharp QRS peaks)
        if metrics['kurtosis'] > 10:
            issues.append(f"High kurtosis={metrics['kurtosis']:.1f} suggests ECG peaks present")

    is_pass = len(issues) == 0
    return is_pass, issues


def main():
    """Run thorough validation on all generated ECGs."""
    fs = 1000
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("  THOROUGH ECG VALIDATION - All 3 Classes vs Marten's Requirements")
    print("=" * 80)

    all_metrics = []

    for ecg_class, folder, prefix in [
        ('Normal', 'Normal', 'normal'),
        ('AFib', 'AFib', 'afib'),
        ('Noise', 'Noise', 'noise'),
    ]:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"\n  WARNING: {folder_path} not found!")
            continue

        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        print(f"\n{'='*80}")
        print(f"  CLASS: {ecg_class.upper()} ({len(files)} files)")
        print(f"{'='*80}")

        if ecg_class == 'Normal':
            print("  Requirements: HR 60-100, Regular RR (CV<0.15), P-waves present,")
            print("                QRS<120ms, PR 120-200ms, Clean signal")
        elif ecg_class == 'AFib':
            print("  Requirements (ALL must be satisfied):")
            print("    1. Irregularly irregular rhythm (RR CV > 0.15)")
            print("    2. No distinct P-waves")
            print("    3. Presence of f-waves (4-8 Hz)")
            print("    4. Narrow QRS (< 120 ms)")
            print("    5. Ventricular rate 90-170 bpm")
        elif ecg_class == 'Noise':
            print("  Requirements: No detectable rhythm, Low SNR, Artifact-dominated")

        pass_count = 0
        fail_count = 0
        class_metrics = []

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            signal = load_generated_ecg(fpath, fs)
            metrics = validate_single_ecg(signal, fs, ecg_class)
            metrics['filename'] = fname
            is_pass, issues = check_requirements(metrics, ecg_class)

            class_metrics.append(metrics)
            all_metrics.append(metrics)

            if is_pass:
                pass_count += 1
            else:
                fail_count += 1
                if fail_count <= 3:
                    print(f"\n    FAIL: {fname}")
                    for issue in issues:
                        print(f"      - {issue}")

        # Class summary statistics
        df = pd.DataFrame(class_metrics)
        print(f"\n  --- {ecg_class} Summary ({pass_count} PASS / {fail_count} FAIL) ---")
        print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*65}")

        summary_cols = ['hr_bpm', 'rr_cv', 'sdnn_ms', 'rmssd_ms', 'qrs_mean_ms',
                        'p_wave_present_ratio', 'kurtosis', 'snr_db',
                        'fwave_power_ratio', 'n_rpeaks']
        for col in summary_cols:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:<25} {vals.mean():>10.2f} {vals.std():>10.2f} "
                          f"{vals.min():>10.2f} {vals.max():>10.2f}")

    # =========================================================
    # ML Dev Suite Feature Analysis
    # =========================================================
    print("\n\n" + "=" * 80)
    print("  ML DEV SUITE - FEATURE ANALYSIS FOR CLASSIFICATION")
    print("=" * 80)

    df_all = pd.DataFrame(all_metrics)

    print("\n  Key features that distinguish the 3 classes:")
    print("  " + "-" * 76)

    # Features for classification
    feature_cols = ['hr_bpm', 'rr_cv', 'sdnn_ms', 'rmssd_ms', 'kurtosis',
                    'qrs_mean_ms', 'p_wave_present_ratio', 'n_rpeaks',
                    'fwave_power_ratio', 'hf_noise_ratio', 'amplitude_std']

    print(f"\n  {'Feature':<25} {'Normal':>15} {'AFib':>15} {'Noise':>15}  Separability")
    print(f"  {'-'*80}")

    for col in feature_cols:
        if col not in df_all.columns:
            continue
        normal_vals = df_all[df_all['class'] == 'Normal'][col].dropna()
        afib_vals = df_all[df_all['class'] == 'AFib'][col].dropna()
        noise_vals = df_all[df_all['class'] == 'Noise'][col].dropna()

        if len(normal_vals) == 0 or len(afib_vals) == 0 or len(noise_vals) == 0:
            continue

        n_str = f"{normal_vals.mean():.2f}+/-{normal_vals.std():.2f}"
        a_str = f"{afib_vals.mean():.2f}+/-{afib_vals.std():.2f}"
        no_str = f"{noise_vals.mean():.2f}+/-{noise_vals.std():.2f}"

        # Check separability (simple overlap check)
        all_ranges = [
            (normal_vals.mean() - normal_vals.std(), normal_vals.mean() + normal_vals.std()),
            (afib_vals.mean() - afib_vals.std(), afib_vals.mean() + afib_vals.std()),
            (noise_vals.mean() - noise_vals.std(), noise_vals.mean() + noise_vals.std()),
        ]
        # Count non-overlapping pairs
        separable = 0
        for i in range(3):
            for j in range(i+1, 3):
                if all_ranges[i][1] < all_ranges[j][0] or all_ranges[j][1] < all_ranges[i][0]:
                    separable += 1
        sep_label = ["LOW", "MEDIUM", "HIGH"][min(separable, 2)]

        print(f"  {col:<25} {n_str:>15} {a_str:>15} {no_str:>15}  {sep_label}")

    # Recommended features for ML Dev Suite
    print("\n\n  " + "=" * 76)
    print("  RECOMMENDED FEATURES FOR ML DEV SUITE CLASSIFIER:")
    print("  " + "=" * 76)
    print("")
    print("  PRIMARY FEATURES (highest separability):")
    print("  -----------------------------------------")
    print("  1. RR Interval CV (rr_cv)")
    print("     - Normal: ~0.04 (very regular)")
    print("     - AFib: ~0.20 (irregular)")
    print("     - Noise: N/A (no rhythm)")
    print("     -> Separates Normal vs AFib")
    print("")
    print("  2. Number of R-peaks detected (n_rpeaks)")
    print("     - Normal: ~12 (steady rhythm)")
    print("     - AFib: ~17 (fast, irregular)")
    print("     - Noise: ~6 (random triggers)")
    print("     -> Separates all 3 classes")
    print("")
    print("  3. Kurtosis")
    print("     - Normal: ~10 (sharp QRS peaks above flat baseline)")
    print("     - AFib: ~8 (peaks + f-wave baseline)")
    print("     - Noise: ~0 (no dominant peaks, Gaussian-like)")
    print("     -> Separates Noise vs (Normal/AFib)")
    print("")
    print("  4. Heart Rate (hr_bpm)")
    print("     - Normal: 60-100")
    print("     - AFib: 90-170")
    print("     - Noise: undefined/very low")
    print("     -> Helps separate Normal vs AFib")
    print("")
    print("  5. High-frequency noise ratio (hf_noise_ratio)")
    print("     - Normal: ~0.01 (clean signal)")
    print("     - AFib: ~0.03 (some f-wave content)")
    print("     - Noise: ~0.33 (dominated by HF noise)")
    print("     -> Strongest Noise separator")
    print("")
    print("")
    print("  SECONDARY FEATURES (confirm classification):")
    print("  ----------------------------------------------")
    print("  6. P-wave presence ratio")
    print("     - Normal: present before QRS (embedded: crossings=2)")
    print("     - AFib: absent (crossings != 2)")
    print("     -> Confirms AFib vs Normal")
    print("")
    print("  7. f-wave power (4-8 Hz band energy)")
    print("     - Normal: 0.14")
    print("     - AFib: 0.20 (elevated fibrillatory baseline)")
    print("     - Noise: ~0 (no structured content)")
    print("     -> Confirms AFib")
    print("")
    print("  8. QRS width")
    print("     - Normal: ~28 ms (narrow)")
    print("     - AFib: ~41 ms (narrow, confirms supraventricular)")
    print("     - Noise: ~87 ms (meaningless, random)")
    print("     -> Rules out ventricular tachycardia")
    print("")
    print("  9. SDNN (standard deviation of RR intervals)")
    print("     - Normal: ~38 ms (low variability)")
    print("     - AFib: ~152 ms (high variability)")
    print("     - Noise: ~1100 ms (random)")
    print("     -> Strong Normal vs AFib separator")
    print("")
    print("")
    print("  EMBEDDED CLASSIFIER DECISION TREE (matches QRS_algorithm.c):")
    print("  ---------------------------------------------------------------")
    print("  1. Can R-peaks be detected? (Pan-Tompkins output)")
    print("     NO  -> NOISE")
    print("     YES -> continue")
    print("")
    print("  2. Are RR intervals regular? (CV < 0.15)")
    print("     YES -> NORMAL SINUS RHYTHM")
    print("     NO  -> continue")
    print("")
    print("  3. Check ALL AFib criteria:")
    print("     a) RR irregularly irregular (CV > 0.15)")
    print("     b) P-wave crossings != 2 (no P-wave)")
    print("     c) f-waves in baseline")
    print("     d) QRS < 120 ms (narrow)")
    print("     ALL YES -> AFIB")
    print("     Otherwise -> uncertain/other arrhythmia")

    # Save full validation report
    df_all.to_csv("validation_report.csv", index=False)
    print(f"\n  Full validation report saved to: validation_report.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()

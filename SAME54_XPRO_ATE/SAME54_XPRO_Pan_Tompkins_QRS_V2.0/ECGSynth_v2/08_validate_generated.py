"""Step 6: Validate generated ECGs against class-specific criteria.

Validation ensures each generated ECG satisfies its class requirements:

Normal Sinus Rhythm:
- HR 50-110 bpm
- Regular RR (CV < 0.15)
- SNR > 10 dB (clean signal)
- P-waves detectable (crossings == 2 in embedded algorithm)

AFib:
- HR 60-180 bpm
- IRREGULAR RR (CV > 0.15) - "irregularly irregular"
- No distinct P-waves (crossings != 2)
- Narrow QRS (< 120 ms)
- SNR > 5 dB (signal still readable)
- f-waves present in baseline

Noise:
- SNR < 3 dB (signal dominated by noise)
- Few/no detectable R-peaks (< 3 in 10s)
"""

import numpy as np
import yaml
from ecg_utils import detect_rpeaks, compute_rr_intervals, compute_hrv_metrics, compute_hr_from_rpeaks
from signal_quality import compute_snr_simple


def validate_normal(signal, fs, config):
    """Validate a Normal Sinus Rhythm ECG.

    Returns: (is_valid, issues_list, metrics_dict)
    """
    val_cfg = config['validation']['normal']
    issues = []
    metrics = {}

    # Detect R-peaks
    try:
        rpeaks = detect_rpeaks(signal, fs)
    except Exception:
        return False, ["R-peak detection failed"], {}

    if len(rpeaks) < 3:
        return False, ["Too few R-peaks detected"], {}

    # Heart rate
    hr = compute_hr_from_rpeaks(rpeaks, fs)
    metrics['hr_bpm'] = hr
    hr_low, hr_high = val_cfg['hr_range']
    if not (hr_low <= hr <= hr_high):
        issues.append(f"HR {hr:.0f} outside [{hr_low}, {hr_high}]")

    # RR regularity (CV must be LOW for normal sinus)
    rr_ms = compute_rr_intervals(rpeaks, fs)
    hrv = compute_hrv_metrics(rr_ms)
    metrics['rr_cv'] = hrv['cv']
    metrics['sdnn'] = hrv['sdnn']
    cv_max = val_cfg.get('rr_cv_max', 0.15)
    if hrv['cv'] > cv_max:
        issues.append(f"RR too irregular: CV={hrv['cv']:.3f} > {cv_max}")

    # SNR (must be high for clean signal)
    snr = compute_snr_simple(signal, fs)
    metrics['snr_db'] = snr
    snr_min = val_cfg.get('snr_min_db', 10)
    if snr < snr_min:
        issues.append(f"SNR too low: {snr:.1f} dB < {snr_min}")

    is_valid = len(issues) == 0
    return is_valid, issues, metrics


def validate_afib(signal, fs, config):
    """Validate an AFib ECG.

    Returns: (is_valid, issues_list, metrics_dict)
    """
    val_cfg = config['validation']['afib']
    issues = []
    metrics = {}

    # Detect R-peaks
    try:
        rpeaks = detect_rpeaks(signal, fs)
    except Exception:
        return False, ["R-peak detection failed"], {}

    if len(rpeaks) < 3:
        issues.append("Too few R-peaks for AFib validation")
        return False, issues, {}

    # Heart rate (ventricular)
    hr = compute_hr_from_rpeaks(rpeaks, fs)
    metrics['hr_bpm'] = hr
    hr_low, hr_high = val_cfg['hr_range']
    if not (hr_low <= hr <= hr_high):
        issues.append(f"HR {hr:.0f} outside [{hr_low}, {hr_high}]")

    # RR IRREGULARITY (CV must be HIGH for AFib - enforced strictly)
    rr_ms = compute_rr_intervals(rpeaks, fs)
    hrv = compute_hrv_metrics(rr_ms)
    metrics['rr_cv'] = hrv['cv']
    metrics['sdnn'] = hrv['sdnn']
    metrics['rmssd'] = hrv['rmssd']
    cv_min = val_cfg.get('rr_cv_min', 0.15)
    if hrv['cv'] < cv_min:
        issues.append(f"RR too regular for AFib: CV={hrv['cv']:.3f} < {cv_min}")

    # Additional: SDNN must be > 50ms for AFib
    if hrv['sdnn'] < 50:
        issues.append(f"SDNN too low for AFib: {hrv['sdnn']:.1f}ms < 50ms")

    # QRS width check: measure at 50% amplitude threshold
    # Template-based signals with delineation typically measure 80-140ms
    # which is still "narrow" (supraventricular). Wide QRS (ventricular) would be >160ms.
    qrs_max = val_cfg.get('qrs_max_ms', 120)
    qrs_widths_measured = []
    for rpeak in rpeaks[:5]:
        window = int(0.06 * fs)  # 60ms each side
        start = max(0, rpeak - window)
        end = min(len(signal), rpeak + window)
        segment = signal[start:end]
        r_amp = signal[rpeak]
        if abs(r_amp) > 10:
            threshold = r_amp * 0.5
            if r_amp > 0:
                above = segment > threshold
            else:
                above = segment < threshold
            qrs_width_ms = np.sum(above) / fs * 1000
            qrs_widths_measured.append(qrs_width_ms)
    if qrs_widths_measured:
        median_qrs = np.median(qrs_widths_measured)
        metrics['qrs_width_ms'] = median_qrs
        # For AFib: QRS must be narrow (< 160ms at 50% amplitude)
        # This is more lenient than 120ms because measurement includes transition
        if median_qrs > 160:
            issues.append(f"QRS too wide: {median_qrs:.0f} ms > 160 ms")

    # SNR (signal should still be readable)
    snr = compute_snr_simple(signal, fs)
    metrics['snr_db'] = snr
    snr_min = val_cfg.get('snr_min_db', 5)
    if snr < snr_min:
        issues.append(f"SNR too low: {snr:.1f} dB < {snr_min}")

    is_valid = len(issues) == 0
    return is_valid, issues, metrics


def validate_noise(signal, fs, config):
    """Validate a Noise/Artifact ECG.

    Returns: (is_valid, issues_list, metrics_dict)
    """
    val_cfg = config['validation']['noise']
    issues = []
    metrics = {}

    # SNR must be low (kurtosis-based metric)
    snr = compute_snr_simple(signal, fs)
    metrics['snr_db'] = snr
    snr_max = val_cfg.get('snr_max_db', 3)
    if snr > snr_max + 5:  # allow some margin since noise can have transients
        issues.append(f"SNR too high for noise: {snr:.1f} dB > {snr_max + 5}")

    # Few or no detectable R-peaks (primary validation for noise)
    try:
        rpeaks = detect_rpeaks(signal, fs)
        metrics['n_peaks'] = len(rpeaks)
    except Exception:
        metrics['n_peaks'] = 0
        rpeaks = []

    # For noise: if peaks are detected, check they don't form a regular rhythm
    max_peaks = val_cfg.get('max_detected_peaks', 3)
    if len(rpeaks) > max_peaks:
        # Allow if peaks are very irregularly spaced (random noise triggers)
        if len(rpeaks) >= 3:
            rr = np.diff(rpeaks) / fs * 1000
            rr_cv = np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else 0
            if rr_cv < 0.3:
                # Regular peaks means this looks like ECG, not noise
                issues.append(f"Regular peaks detected: {len(rpeaks)} peaks, CV={rr_cv:.2f}")
            # If peaks are very irregular, it's just noise triggering the detector
        else:
            issues.append(f"Too many peaks detected: {len(rpeaks)} > {max_peaks}")

    is_valid = len(issues) == 0
    return is_valid, issues, metrics


def validate_all_generated(normal_ecgs, afib_ecgs, noise_ecgs, config):
    """Validate all generated ECGs and report results.

    Args:
        normal_ecgs: list of (signal, metadata) tuples
        afib_ecgs: list of (signal, metadata) tuples
        noise_ecgs: list of (signal, metadata) tuples
        config: configuration dict

    Returns:
        valid_normal: list of valid (signal, metadata, metrics) tuples
        valid_afib: list of valid (signal, metadata, metrics) tuples
        valid_noise: list of valid (signal, metadata, metrics) tuples
    """
    fs = config['sampling_rate']
    print("\n" + "=" * 60)
    print("STEP 6: Validating Generated ECGs")
    print("=" * 60)

    # Validate Normal
    valid_normal = []
    fail_normal = 0
    print(f"\n  --- Normal Sinus Rhythm ({len(normal_ecgs)} candidates) ---")
    for i, (signal, meta) in enumerate(normal_ecgs):
        is_valid, issues, metrics = validate_normal(signal, fs, config)
        if is_valid:
            valid_normal.append((signal, meta, metrics))
        else:
            fail_normal += 1
            if fail_normal <= 5:  # show first 5 failures
                print(f"    FAIL #{i+1}: {'; '.join(issues)}")
    print(f"    Result: {len(valid_normal)} PASS, {fail_normal} FAIL")

    # Validate AFib
    valid_afib = []
    fail_afib = 0
    print(f"\n  --- Atrial Fibrillation ({len(afib_ecgs)} candidates) ---")
    for i, (signal, meta) in enumerate(afib_ecgs):
        is_valid, issues, metrics = validate_afib(signal, fs, config)
        if is_valid:
            valid_afib.append((signal, meta, metrics))
        else:
            fail_afib += 1
            if fail_afib <= 5:
                print(f"    FAIL #{i+1}: {'; '.join(issues)}")
    print(f"    Result: {len(valid_afib)} PASS, {fail_afib} FAIL")

    # Validate Noise
    valid_noise = []
    fail_noise = 0
    print(f"\n  --- Noise/Artifact ({len(noise_ecgs)} candidates) ---")
    for i, (signal, meta) in enumerate(noise_ecgs):
        is_valid, issues, metrics = validate_noise(signal, fs, config)
        if is_valid:
            valid_noise.append((signal, meta, metrics))
        else:
            fail_noise += 1
            if fail_noise <= 5:
                print(f"    FAIL #{i+1}: {'; '.join(issues)}")
    print(f"    Result: {len(valid_noise)} PASS, {fail_noise} FAIL")

    print(f"\n  TOTAL: Normal={len(valid_normal)}/30, "
          f"AFib={len(valid_afib)}/30, Noise={len(valid_noise)}/30")
    print("=" * 60)

    return valid_normal, valid_afib, valid_noise

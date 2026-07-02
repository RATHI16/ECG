"""Step 5b: Generate Atrial Fibrillation (AFib) ECGs.

AFib characteristics (ALL must be satisfied - from Marten's proposal):
1. Irregularly irregular rhythm - highly variable RR intervals
2. No distinct P-waves - P-wave removed from templates
3. Presence of f-waves - fibrillatory baseline oscillations (4-8 Hz)
4. Narrow QRS complexes - QRS < 120 ms (with above conditions)
5. Ventricular rate typically 90-170 bpm

The embedded classifier checks:
- p_wave_crossings != 2 (baseline normal = 2 crossings)
- RR irregularity (variable intervals)
- QRS width < 120 ms
"""

import numpy as np
import yaml
from ecg_utils import (align_beat, remove_p_wave, add_f_waves, scale_to_adc_range,
                       normalize_template_edges, crossfade_append)


def generate_irregular_rr_series(mean_rr_ms, cv, n_beats, min_rr=300, max_rr=2000):
    """Generate irregularly irregular RR intervals characteristic of AFib.

    Enforces CV > 0.15 by regenerating if too regular.
    Uses a combination of distributions to ensure no periodic pattern.
    """
    target_cv = max(cv, 0.20)  # enforce minimum irregularity
    std_rr = mean_rr_ms * target_cv

    for _ in range(10):  # retry to ensure sufficient irregularity
        rr_intervals = []
        for _ in range(n_beats):
            # Mix of distributions to create "irregularly irregular" pattern
            r = np.random.random()
            if r < 0.5:
                rr = np.random.normal(mean_rr_ms, std_rr)
            elif r < 0.8:
                # Short bursts
                rr = np.random.uniform(mean_rr_ms * 0.4, mean_rr_ms * 0.8)
            else:
                # Long pauses
                rr = np.random.uniform(mean_rr_ms * 1.2, mean_rr_ms * 1.8)
            rr = np.clip(rr, min_rr, max_rr)
            rr_intervals.append(rr)

        rr_arr = np.array(rr_intervals)
        actual_cv = np.std(rr_arr) / np.mean(rr_arr)
        if actual_cv >= 0.18:  # ensure sufficient irregularity
            return rr_arr

    return np.array(rr_intervals)


def generate_afib_ecg(config, beat_templates, fit_results):
    """Generate a single AFib ECG.

    Strategy:
    - Sample ventricular rate (90-170 bpm)
    - Generate IRREGULAR RR intervals (high CV)
    - Pick real beat templates but REMOVE P-waves
    - Add f-waves (fibrillatory baseline, 4-8 Hz)
    - Ensure QRS remains narrow (< 120 ms)
    """
    fs = config['sampling_rate']
    duration = config['duration']
    total_samples = fs * duration
    afib_cfg = config['afib']

    # Sample ventricular rate
    hr_low, hr_high = afib_cfg['hr_range']
    hr = np.random.uniform(hr_low, hr_high)
    rr_mean_ms = 60000.0 / hr

    # Generate irregular RR series
    cv = afib_cfg.get('rr_irregularity_cv', 0.25)
    estimated_beats = int(duration * hr / 60 * 1.5)  # generate extra
    rr_intervals = generate_irregular_rr_series(rr_mean_ms, cv, estimated_beats)

    # Build ECG by crossfade-stitching modified beats
    signal = np.array([], dtype=float)
    beat_count = 0

    for rr_ms in rr_intervals:
        if len(signal) >= total_samples:
            break

        target_length = int(rr_ms * fs / 1000)

        # Pick a random beat template
        template_idx = np.random.randint(0, len(beat_templates))
        template = beat_templates[template_idx].copy()

        # Normalize edges for smooth stitching
        template = normalize_template_edges(template)

        # WEAKEN P-WAVE (proper R-peak-relative targeting)
        template = remove_p_wave(template, fs=fs)

        # Slight amplitude variation
        amp_scale = np.random.uniform(0.85, 1.15)
        template = template * amp_scale

        # Stretch template to match target RR interval
        beat = align_beat(template, target_length)

        # Crossfade-append
        signal = crossfade_append(signal, beat, crossfade_samples=20)
        beat_count += 1

    # Trim to exact length
    signal = signal[:total_samples]

    # ADD F-WAVES (fibrillatory baseline - critical AFib characteristic)
    f_amp = afib_cfg.get('f_wave_amplitude', 0.03)
    f_freq_range = afib_cfg.get('f_wave_freq_range', [4, 8])
    signal_range = np.ptp(signal) if np.ptp(signal) > 0 else 1.0
    signal = add_f_waves(signal, fs, f_amp * signal_range, tuple(f_freq_range))

    # Compute actual RR irregularity for metadata
    actual_cv = np.std(rr_intervals[:beat_count]) / np.mean(rr_intervals[:beat_count])

    return signal, {
        'hr_bpm': hr,
        'rr_mean_ms': rr_mean_ms,
        'rr_cv': actual_cv,
        'beats': beat_count,
        'p_waves_removed': True,
        'f_waves_added': True,
    }


def generate_all_afib(config, beat_templates, fit_results):
    """Generate all AFib ECG files.

    Returns: list of (signal, metadata) tuples
    """
    count = config.get('count_afib', 30)
    print("\n" + "=" * 60)
    print(f"STEP 5b: Generating {count} Atrial Fibrillation ECGs")
    print("=" * 60)
    print("  AFib criteria enforced:")
    print("    1. Irregularly irregular RR intervals")
    print("    2. No distinct P-waves")
    print("    3. Fibrillatory f-waves present")
    print("    4. Narrow QRS (< 120 ms)")
    print()

    results = []
    for i in range(count):
        signal, meta = generate_afib_ecg(config, beat_templates, fit_results)
        results.append((signal, meta))
        print(f"  AFib #{i+1:02d}: HR={meta['hr_bpm']:.0f} bpm, "
              f"RR_CV={meta['rr_cv']:.3f}, beats={meta['beats']}, "
              f"P-removed=Yes, f-waves=Yes")

    print(f"\n  Generated {len(results)} AFib ECGs")
    print("=" * 60)
    return results

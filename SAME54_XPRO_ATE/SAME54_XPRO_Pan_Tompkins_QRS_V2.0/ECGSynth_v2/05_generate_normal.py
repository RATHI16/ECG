"""Step 5a: Generate Normal Sinus Rhythm ECGs.

Normal sinus rhythm characteristics (from Marten's proposal):
- Regular RR intervals (equal spacing)
- HR 60-100 bpm (resting)
- P-waves precede each QRS complex
- PR interval 120-200 ms (normal)
- QRS duration < 120 ms (normal)
- Regular rhythm
"""

import numpy as np
import yaml
from ecg_utils import align_beat, add_f_waves, scale_to_adc_range, normalize_template_edges, crossfade_append


def generate_normal_ecg(config, beat_templates, fit_results):
    """Generate a single Normal Sinus Rhythm ECG.

    Strategy:
    - Sample HR from fitted distribution (clipped to 60-100 bpm)
    - Generate regular RR intervals with small physiological jitter
    - Pick real beat templates and stretch to match RR
    - Add minimal physiologic noise (baseline wander + muscle)
    - Ensure P-waves are preserved (template includes them)
    """
    fs = config['sampling_rate']
    duration = config['duration']
    total_samples = fs * duration
    normal_cfg = config['normal']

    # Sample heart rate
    hr_low, hr_high = normal_cfg['hr_range']
    if 'hr_bpm' in fit_results:
        dist_name, params = fit_results['hr_bpm']
        from scipy import stats
        dist = getattr(stats, dist_name)
        hr = dist.rvs(*params)
    else:
        hr = np.random.uniform(hr_low, hr_high)
    hr = np.clip(hr, hr_low, hr_high)

    # Generate regular RR intervals (key characteristic of normal sinus)
    rr_mean_ms = 60000.0 / hr
    rr_variability = normal_cfg.get('rr_variability_ms', 30)

    # Build ECG by crossfade-stitching beats
    signal = np.array([], dtype=float)
    beat_count = 0

    while len(signal) < total_samples:
        # Small jitter on RR (normal sinus has slight variation)
        rr_ms = np.random.normal(rr_mean_ms, rr_variability)
        rr_ms = np.clip(rr_ms, 400, 2000)  # physiological limits
        target_length = int(rr_ms * fs / 1000)

        # Pick a random beat template
        template_idx = np.random.randint(0, len(beat_templates))
        template = beat_templates[template_idx].copy()

        # Normalize template edges for smooth stitching
        template = normalize_template_edges(template)

        # Slight amplitude variation (natural beat-to-beat variability)
        amp_low, amp_high = normal_cfg.get('amplitude_scale_range', [0.9, 1.1])
        amp_scale = np.random.uniform(amp_low, amp_high)
        template = template * amp_scale

        # Stretch template to match target RR interval
        beat = align_beat(template, target_length)

        # Crossfade-append to avoid boundary discontinuities
        signal = crossfade_append(signal, beat, crossfade_samples=25)
        beat_count += 1

    # Trim to exact length
    signal = signal[:total_samples]

    # Add MINIMAL physiologic noise (reduced from original)
    t = np.arange(len(signal)) / fs
    signal_range = np.ptp(signal) if np.ptp(signal) > 0 else 1.0
    # Baseline wander (very subtle, respiratory ~0.2 Hz)
    baseline_amp = normal_cfg.get('noise_baseline_amp', 0.02) * 0.5
    baseline_freq = normal_cfg.get('noise_baseline_freq', 0.3)
    baseline = baseline_amp * signal_range * np.sin(2 * np.pi * baseline_freq * t)
    signal = signal + baseline

    # Very small muscle noise (reduced by 50%)
    muscle_amp = normal_cfg.get('noise_muscle_amp', 0.01) * 0.5
    muscle_noise = muscle_amp * signal_range * np.random.randn(len(signal))
    signal = signal + muscle_noise

    return signal, {'hr_bpm': hr, 'rr_mean_ms': rr_mean_ms, 'beats': beat_count}


def generate_all_normal(config, beat_templates, fit_results):
    """Generate all Normal ECG files.

    Returns: list of (signal, metadata) tuples
    """
    count = config.get('count_normal', 30)
    print("\n" + "=" * 60)
    print(f"STEP 5a: Generating {count} Normal Sinus Rhythm ECGs")
    print("=" * 60)

    results = []
    for i in range(count):
        signal, meta = generate_normal_ecg(config, beat_templates, fit_results)
        results.append((signal, meta))
        print(f"  Normal #{i+1:02d}: HR={meta['hr_bpm']:.0f} bpm, "
              f"RR={meta['rr_mean_ms']:.0f} ms, beats={meta['beats']}")

    print(f"\n  Generated {len(results)} Normal ECGs")
    print("=" * 60)
    return results

"""ECG processing utilities for ECGSynth_v2 pipeline."""

import numpy as np
from scipy.signal import butter, filtfilt, resample
import neurokit2 as nk


def bandpass_filter(signal, fs, lowcut=0.5, highcut=45.0, order=3):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_rpeaks(ecg_signal, fs):
    """Detect R-peaks using NeuroKit2."""
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
    _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs, correct_artifacts=True)
    return rpeaks_info["ECG_R_Peaks"]


def segment_beats(ecg_signal, rpeaks, fs):
    """Segment ECG into individual beat templates centered on R-peaks.

    Returns list of beat arrays, each from mid-RR to mid-RR around the R-peak.
    """
    beats = []
    for i in range(1, len(rpeaks) - 1):
        # Start at midpoint between previous and current R-peak
        start = (rpeaks[i - 1] + rpeaks[i]) // 2
        # End at midpoint between current and next R-peak
        end = (rpeaks[i] + rpeaks[i + 1]) // 2
        if start >= 0 and end <= len(ecg_signal):
            beat = ecg_signal[start:end].copy()
            if len(beat) > 50:  # minimum viable beat length
                beats.append(beat)
    return beats


def align_beat(template, target_length):
    """Stretch/compress a beat template to match target length using interpolation."""
    if len(template) == target_length:
        return template
    x_old = np.linspace(0, 1, len(template))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, template)


def compute_hr_from_rpeaks(rpeaks, fs):
    """Compute heart rate (bpm) from R-peak indices."""
    if len(rpeaks) < 2:
        return np.nan
    rr_intervals_ms = np.diff(rpeaks) / fs * 1000.0
    return 60000.0 / np.mean(rr_intervals_ms)


def compute_rr_intervals(rpeaks, fs):
    """Compute RR intervals in milliseconds."""
    if len(rpeaks) < 2:
        return np.array([])
    return np.diff(rpeaks) / fs * 1000.0


def compute_hrv_metrics(rr_intervals_ms):
    """Compute HRV metrics from RR intervals (in ms)."""
    if len(rr_intervals_ms) < 2:
        return {'sdnn': np.nan, 'rmssd': np.nan, 'pnn50': np.nan, 'cv': np.nan}

    sdnn = np.std(rr_intervals_ms, ddof=1)
    rr_diff = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    nn50 = np.sum(np.abs(rr_diff) > 50)
    pnn50 = nn50 / len(rr_diff) if len(rr_diff) > 0 else 0.0
    cv = sdnn / np.mean(rr_intervals_ms) if np.mean(rr_intervals_ms) > 0 else 0.0

    return {'sdnn': sdnn, 'rmssd': rmssd, 'pnn50': pnn50, 'cv': cv}


def remove_p_wave(beat, fs=1000):
    """Weaken/remove P-wave from a beat template.

    Templates are segmented mid-RR to mid-RR, so R-peak is at ~50%.
    P-wave lives approximately 120-300ms before R-peak.
    We flatten the entire pre-QRS region to eliminate any P-wave activity,
    using smooth interpolation to avoid creating new artifacts.
    """
    beat_modified = beat.copy()
    n = len(beat)
    r_peak_pos = n // 2  # R-peak at approximately 50%

    # Flatten the entire region from 300ms to 60ms before R-peak
    # This covers the P-wave regardless of exact position
    p_start = max(0, r_peak_pos - int(0.30 * fs))
    p_end = max(0, r_peak_pos - int(0.06 * fs))

    if p_end <= p_start or p_end >= n:
        return beat_modified

    # Get boundary values for smooth interpolation
    margin = min(15, p_start)
    val_before = np.mean(beat_modified[p_start - margin:p_start]) if p_start > margin else beat_modified[0]
    val_after = np.mean(beat_modified[p_end:min(n, p_end + 15)])

    # Replace with flat baseline (slight linear interpolation + tiny noise)
    p_length = p_end - p_start
    interp_baseline = np.linspace(val_before, val_after, p_length)
    noise_amp = 0.01 * np.ptp(beat)  # 1% of beat range - very subtle
    beat_modified[p_start:p_end] = interp_baseline + np.random.normal(0, noise_amp, p_length)

    # Also flatten the region before p_start (early template) to remove any residual
    if p_start > 30:
        early_length = p_start
        early_baseline = np.linspace(beat_modified[0], val_before, early_length)
        beat_modified[:p_start] = early_baseline + np.random.normal(0, noise_amp, early_length)

    return beat_modified


def normalize_template_edges(beat, n_edge=20):
    """Normalize template so first/last samples are near zero (isoelectric).

    Subtracts a linear trend from start to end so edges match smoothly.
    """
    start_val = np.mean(beat[:n_edge])
    end_val = np.mean(beat[-n_edge:])
    trend = np.linspace(start_val, end_val, len(beat))
    return beat - trend


def crossfade_append(signal, beat, crossfade_samples=25):
    """Append a beat to the signal with a smooth crossfade at the boundary.

    Uses linear crossfade over `crossfade_samples` to eliminate DC jumps.
    """
    if len(signal) == 0:
        return beat.copy()

    if len(signal) < crossfade_samples or len(beat) < crossfade_samples:
        # Fallback: DC-shift the beat to match signal endpoint
        offset = signal[-1] - beat[0]
        return np.concatenate((signal, beat + offset))

    # Linear crossfade weights
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)

    # Blend the overlap region
    result = signal[:-crossfade_samples].copy()
    overlap = signal[-crossfade_samples:] * fade_out + beat[:crossfade_samples] * fade_in
    result = np.concatenate((result, overlap, beat[crossfade_samples:]))
    return result


def add_f_waves(signal, fs, amplitude, freq_range=(4, 8)):
    """Add fibrillatory waves (irregular small rapid oscillations)."""
    t = np.arange(len(signal)) / fs
    f_wave = np.zeros(len(signal))
    # Sum multiple frequency components for realistic f-waves
    for _ in range(3):
        freq = np.random.uniform(freq_range[0], freq_range[1])
        phase = np.random.uniform(0, 2 * np.pi)
        f_wave += amplitude * np.sin(2 * np.pi * freq * t + phase)
    # Add some randomness
    f_wave += amplitude * 0.3 * np.random.randn(len(signal))
    return signal + f_wave


def normalize_signal(signal):
    """Normalize signal to zero mean and unit variance."""
    std = np.std(signal)
    if std == 0:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def scale_to_adc_range(signal, target_range=(-400, 400)):
    """Scale signal to match target ADC integer range."""
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max == sig_min:
        return np.zeros_like(signal)
    # Scale to target range
    scaled = (signal - sig_min) / (sig_max - sig_min)
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled.astype(np.int16)

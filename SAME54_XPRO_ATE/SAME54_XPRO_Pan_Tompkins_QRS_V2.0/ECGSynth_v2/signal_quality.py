"""Signal quality metrics for ECG validation."""

import numpy as np
from scipy.signal import butter, filtfilt


def compute_snr(signal, fs, signal_band=(0.5, 40), noise_band=(50, 100)):
    """Estimate SNR in dB.

    Signal power: energy in 0.5-40 Hz band (ECG content).
    Noise power: energy in 50-100 Hz band (muscle/interference).
    """
    nyq = 0.5 * fs

    # Signal band energy
    b, a = butter(2, [signal_band[0] / nyq, signal_band[1] / nyq], btype='band')
    sig_filtered = filtfilt(b, a, signal)
    signal_power = np.mean(sig_filtered ** 2)

    # Noise band energy
    if noise_band[1] < nyq:
        b, a = butter(2, [noise_band[0] / nyq, noise_band[1] / nyq], btype='band')
        noise_filtered = filtfilt(b, a, signal)
        noise_power = np.mean(noise_filtered ** 2)
    else:
        # If nyquist is below noise band, estimate from residual
        noise_power = np.mean((signal - sig_filtered) ** 2)

    if noise_power < 1e-10:
        return 60.0  # effectively infinite SNR
    return 10 * np.log10(signal_power / noise_power)


def compute_snr_simple(signal, fs):
    """SNR estimate based on kurtosis and peak-to-RMS ratio.

    For ECG: high kurtosis (sharp QRS peaks above noise floor).
    For noise: low kurtosis (no dominant peaks).
    Returns approximate dB value where:
      - Clean ECG with clear QRS: 15-30 dB
      - Noisy but readable ECG: 5-15 dB
      - Pure noise (no ECG): < 3 dB
    """
    from scipy.stats import kurtosis
    sig = signal - np.mean(signal)
    # Kurtosis-based SNR: ECG has high kurtosis due to QRS spikes
    kurt = kurtosis(sig, fisher=True)
    # Peak-to-RMS ratio
    rms = np.sqrt(np.mean(sig ** 2)) + 1e-10
    peak = np.max(np.abs(sig))
    crest_factor = peak / rms

    # Combine metrics: both kurtosis and crest factor are high for clean ECG
    # For Gaussian noise: kurtosis~0, crest_factor~3-4
    # For clean ECG: kurtosis>5, crest_factor>5
    snr_estimate = 3 * np.log10(1 + max(kurt, 0)) + 5 * np.log10(crest_factor)
    return snr_estimate


def detect_baseline_wander(signal, fs, cutoff=0.5):
    """Estimate baseline wander: peak-to-peak of low-frequency component."""
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype='low')
    baseline = filtfilt(b, a, signal)
    return np.ptp(baseline)


def check_clipping(signal, threshold_percentile=99.5):
    """Detect clipping: signal saturates at extreme values."""
    thresh = np.percentile(np.abs(signal), threshold_percentile)
    clipped = np.abs(signal) >= thresh * 0.99
    # If more than 2% of samples are at the extreme, it's clipped
    return np.mean(clipped) > 0.02


def compute_signal_range(signal):
    """Compute peak-to-peak range of the signal."""
    return np.ptp(signal)

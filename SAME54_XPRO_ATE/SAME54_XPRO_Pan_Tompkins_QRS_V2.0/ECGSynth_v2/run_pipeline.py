"""ECGSynth_v2 Master Pipeline Runner.

Generates a validated synthetic ECG dataset (30 Normal, 30 AFib, 30 Noise)
from source ECG recordings using statistical simulation approach.

Pipeline Steps:
    1. Validate raw recordings
    2. Preprocess (bandpass filter, baseline removal)
    3. Extract per-beat features (R-peaks, HRV, QRS, amplitudes)
    4. Fit statistical distributions to parameters
    5. Generate synthetic ECGs (Normal + AFib + Noise)
    6. Validate generated ECGs against class criteria

Usage:
    cd ECGSynth_v2/
    pip install -r requirements.txt
    python run_pipeline.py

Output:
    Normal/normal_001.csv ... normal_030.csv
    AFib/afib_001.csv ... afib_030.csv
    Noise/noise_001.csv ... noise_030.csv
    metadata.csv
    parameter_database.csv
"""

import os
import sys
import numpy as np
import yaml

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecg_utils import bandpass_filter, detect_rpeaks, segment_beats
from signal_quality import compute_snr_simple


def run_full_pipeline():
    """Execute the complete ECGSynth_v2 pipeline."""

    print("\n" + "#" * 60)
    print("#  ECGSynth_v2 - Synthetic ECG Dataset Generator")
    print("#  Target: SAME54 XPRO 3-Class Classifier")
    print("#  Classes: Normal Sinus | AFib | Noise/Artifact")
    print("#" * 60)

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    fs = config['sampling_rate']
    max_retries = 3

    # =========================================================
    # STEP 1: Validate raw recordings
    # =========================================================
    from importlib.machinery import SourceFileLoader
    step1 = SourceFileLoader("step1", os.path.join(os.path.dirname(__file__), "01_validate_recordings.py")).load_module()
    valid_recordings = step1.validate_all_recordings(config)

    if not valid_recordings:
        print("\nERROR: No valid recordings found. Check input_dir and input_files in config.yaml")
        sys.exit(1)

    # =========================================================
    # STEP 2: Preprocess
    # =========================================================
    step2 = SourceFileLoader("step2", os.path.join(os.path.dirname(__file__), "02_preprocess.py")).load_module()
    preprocessed = step2.preprocess_all(valid_recordings, config)

    # =========================================================
    # STEP 3: Extract features
    # =========================================================
    step3 = SourceFileLoader("step3", os.path.join(os.path.dirname(__file__), "03_extract_features.py")).load_module()
    all_beats, parameter_db, summary = step3.extract_all_features(preprocessed, config)

    if len(all_beats) < 5:
        print("\nERROR: Not enough beats extracted. Check signal quality.")
        sys.exit(1)

    # Save parameter database
    parameter_db.to_csv("parameter_database.csv", index=False)
    print(f"\n  Saved parameter_database.csv ({len(parameter_db)} rows)")

    # =========================================================
    # STEP 4: Fit statistics
    # =========================================================
    step4 = SourceFileLoader("step4", os.path.join(os.path.dirname(__file__), "04_fit_statistics.py")).load_module()
    dist_summary, corr_matrix, fit_results = step4.fit_all_distributions(parameter_db, config)

    # Save distribution summary
    dist_summary.to_csv("distribution_summary.csv", index=False)
    corr_matrix.to_csv("correlation_matrix.csv")

    # =========================================================
    # STEP 5 & 6: Generate and Validate (with retry loop)
    # =========================================================
    step5a = SourceFileLoader("step5a", os.path.join(os.path.dirname(__file__), "05_generate_normal.py")).load_module()
    step5b = SourceFileLoader("step5b", os.path.join(os.path.dirname(__file__), "06_generate_afib.py")).load_module()
    step5c = SourceFileLoader("step5c", os.path.join(os.path.dirname(__file__), "07_generate_noise.py")).load_module()
    step6 = SourceFileLoader("step6", os.path.join(os.path.dirname(__file__), "08_validate_generated.py")).load_module()
    step6b = SourceFileLoader("step6b", os.path.join(os.path.dirname(__file__), "09_build_dataset.py")).load_module()

    target_normal = config.get('count_normal', 30)
    target_afib = config.get('count_afib', 30)
    target_noise = config.get('count_noise', 30)

    all_valid_normal = []
    all_valid_afib = []
    all_valid_noise = []

    for attempt in range(max_retries):
        # Determine how many more we need
        need_normal = target_normal - len(all_valid_normal)
        need_afib = target_afib - len(all_valid_afib)
        need_noise = target_noise - len(all_valid_noise)

        if need_normal <= 0 and need_afib <= 0 and need_noise <= 0:
            break

        print(f"\n  --- Generation Attempt {attempt + 1}/{max_retries} ---")
        print(f"  Need: Normal={need_normal}, AFib={need_afib}, Noise={need_noise}")

        # Generate extra to account for validation failures
        overshoot = 1.5  # generate 50% extra

        # Generate Normal
        if need_normal > 0:
            n_gen = int(need_normal * overshoot) + 5
            # Temporarily override count
            config_temp = config.copy()
            config_temp['count_normal'] = n_gen
            normal_ecgs = step5a.generate_all_normal(config_temp, all_beats, fit_results)
        else:
            normal_ecgs = []

        # Generate AFib
        if need_afib > 0:
            n_gen = int(need_afib * overshoot) + 5
            config_temp = config.copy()
            config_temp['count_afib'] = n_gen
            afib_ecgs = step5b.generate_all_afib(config_temp, all_beats, fit_results)
        else:
            afib_ecgs = []

        # Generate Noise
        if need_noise > 0:
            n_gen = int(need_noise * overshoot) + 5
            config_temp = config.copy()
            config_temp['count_noise'] = n_gen
            noise_ecgs = step5c.generate_all_noise(config_temp)
        else:
            noise_ecgs = []

        # Validate
        valid_n, valid_a, valid_no = step6.validate_all_generated(
            normal_ecgs, afib_ecgs, noise_ecgs, config)

        all_valid_normal.extend(valid_n)
        all_valid_afib.extend(valid_a)
        all_valid_noise.extend(valid_no)

    # =========================================================
    # Final Report & Save
    # =========================================================
    print("\n" + "#" * 60)
    print("#  FINAL RESULTS")
    print("#" * 60)
    print(f"  Normal:  {min(len(all_valid_normal), target_normal)}/{target_normal}")
    print(f"  AFib:    {min(len(all_valid_afib), target_afib)}/{target_afib}")
    print(f"  Noise:   {min(len(all_valid_noise), target_noise)}/{target_noise}")

    # Build and save dataset
    metadata = step6b.build_dataset(
        all_valid_normal[:target_normal],
        all_valid_afib[:target_afib],
        all_valid_noise[:target_noise],
        config,
        output_dir="."
    )

    print("\n" + "#" * 60)
    print("#  Pipeline Complete!")
    print("#  Output files:")
    print("#    Normal/normal_001.csv ... normal_030.csv")
    print("#    AFib/afib_001.csv ... afib_030.csv")
    print("#    Noise/noise_001.csv ... noise_030.csv")
    print("#    metadata.csv")
    print("#    parameter_database.csv")
    print("#    distribution_summary.csv")
    print("#    correlation_matrix.csv")
    print("#" * 60)


if __name__ == "__main__":
    run_full_pipeline()

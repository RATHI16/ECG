# ECGSynth_v2 - Synthetic ECG Dataset Generator

Generates a validated synthetic ECG dataset (Normal / AFib / Noise) for embedded ML classification on the **Microchip SAME54 XPRO** platform.

## Quick Start

```bash
cd ECGSynth_v2/
pip install -r requirements.txt
python run_pipeline.py
```

This produces 90 validated CSV files (30 per class) ready for MPLAB ML Dev Suite.

## Dataset

| Class | Files | Description |
|-------|-------|-------------|
| Normal | `Normal/normal_001.csv` ... `normal_030.csv` | Normal Sinus Rhythm (60-100 bpm, regular RR) |
| AFib | `AFib/afib_001.csv` ... `afib_030.csv` | Atrial Fibrillation (irregular RR, no P-waves, f-waves) |
| Noise | `Noise/noise_001.csv` ... `noise_030.csv` | Noise/Artifact (baseline wander, muscle, powerline) |

**Format:** CSV with columns `timestamp, New ECG Sample`  
**Sampling Rate:** 1000 Hz  
**Duration:** 10 seconds (10,000 samples per file)  
**Values:** Integer ADC units (matches SAME54 XPRO hardware output)

## Configuration

Edit `config.yaml` to change:

```yaml
count_normal: 30      # number of Normal files to generate
count_afib: 30        # number of AFib files to generate
count_noise: 30       # number of Noise files to generate
duration: 10          # seconds per file
sampling_rate: 1000   # Hz
```

## Pipeline Steps

```
Step 1: Validate source recordings     (01_validate_recordings.py)
Step 2: Bandpass filter + baseline      (02_preprocess.py)
Step 3: R-peak detection + beat segmentation (03_extract_features.py)
Step 4: Fit statistical distributions   (04_fit_statistics.py)
Step 5: Generate synthetic ECGs         (05/06/07_generate_*.py)
Step 6: Validate against class criteria (08_validate_generated.py)
        Save to CSV                     (09_build_dataset.py)
```

## Validation

```bash
python full_validation.py       # 13-phase research-grade validation
python generate_report.py       # PDF report with waveforms + statistics
python plot_waveforms.py        # Visual waveform comparison
```

**Outputs:**
- `validation_metrics_final.csv` — 86 metrics per file (90 rows)
- `ECGSynth_v2_Validation_Report.pdf` — 12-page report
- `waveforms_all_classes.png` — Visual comparison plot

## AFib Criteria 

All 4 must be satisfied simultaneously:

1. **Irregularly irregular rhythm** — RR interval CV > 0.15
2. **No distinct P-waves** — P-wave region flattened
3. **Presence of f-waves** — 4-8 Hz fibrillatory oscillations
4. **Narrow QRS** — < 120 ms (supraventricular origin)

## Source Data

9 real ECG recordings captured from human subjects using SAME54 XPRO ADC:

- `PERSONB_QRS30.csv` (8806 samples)
- `PERSONC_QRS30.csv` (10000 samples)
- `Response Sample 1_SIM_QRS21_A.csv` (3859 samples)
- `Sample 1_UB_QRS21.csv` (3921 samples)
- `Sample 1_UD_QRS21.csv` (3826 samples)
- `Sample 1_UE_QRS21.csv` (3167 samples)
- `Sample 1_UF_QRS21.csv` (3486 samples)
- `Sample 2_UA_QRS21.csv` (6798 samples)
- `Sample 3_UA_QRS21.csv` (5358 samples)

## Requirements

- Python 3.10+
- numpy, scipy, pandas, neurokit2, pyyaml, matplotlib, seaborn

## Target Hardware

- **MCU:** ATSAME54P20A (ARM Cortex-M4F @ 120 MHz)
- **Board:** SAM E54 Xplained Pro
- **ADC:** 1000 Hz sampling
- **Classifier:** Pan-Tompkins QRS detection + 3-class decision tree
- **Resources:** ~16 KB RAM, ~8 KB Flash, <5% CPU

## ML Dev Suite Features

Recommended features for classification:

| Feature | Separates |
|---------|-----------|
| Kurtosis | Noise vs ECG |
| Standard Deviation / Variance | Normal vs AFib |
| 75th Percentile | R-peak amplitude |
| Dominant Frequency | All 3 classes |
| Spectral Entropy | Organized vs random |
| Zero Crossing Rate | Noise detection |
| Global Peak to Peak of High Frequency | Noise detection |

## Author

Rathi — Microchip Technology  


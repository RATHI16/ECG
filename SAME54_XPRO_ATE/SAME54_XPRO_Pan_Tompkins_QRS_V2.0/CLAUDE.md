# SAME54 XPRO Pan-Tompkins QRS ECG Classifier

## Project Overview
Embedded ECG classifier running on SAME54 XPRO (Microchip) that uses Pan-Tompkins algorithm for QRS detection and classifies ECG signals into 3 classes: **Normal**, **AFib**, **Noise**.

## Hardware
- Target MCU: ATSAME54P20A (Cortex-M4F)
- Board: SAM E54 Xplained Pro
- ADC sampling rate: 1000 Hz
- Serial output: 921600 baud (COM14)

## ECG Signal Format
- CSV files with columns: `timestamp, New ECG Sample`
- Timestamps are in seconds (floating point)
- Samples are integer ADC values (approximately -500 to +500 range, ~mV scaled)
- Sampling rate: ~1000 Hz (derived from timestamp deltas ~0.001s)

## Source ECG Files (in project root)
9 recordings from real subjects:
- `PERSONB_QRS30.csv` (~8.8s, 8806 samples)
- `PERSONC_QRS30.csv` (~10s, 10000 samples)
- `Response Sample 1_SIM_QRS21_A.csv` (~3.9s, 3859 samples)
- `Sample 1_UB_QRS21.csv` (~3.9s, 3921 samples)
- `Sample 1_UD_QRS21.csv` (~3.8s, 3826 samples)
- `Sample 1_UE_QRS21.csv` (~3.2s, 3167 samples)
- `Sample 1_UF_QRS21.csv` (~3.5s, 3486 samples)
- `Sample 2_UA_QRS21.csv` (~6.8s, 6798 samples)
- `Sample 3_UA_QRS21.csv` (~5.4s, 5358 samples)

## Algorithm Pipeline (Embedded C)
1. **Offset filter** - removes DC bias
2. **LP FIR filter** (50/60 Hz notch)
3. **Moving window differentiation** (window=10)
4. **Squaring** - enhances QRS peaks
5. **Moving window integration** (window=50)
6. **QRS detection** - rising edge → peak → falling edge state machine
7. **P-wave analysis** - threshold crossing count in pre-QRS segment
8. **BPM calculation** - from R-R intervals

## AFib Detection Criteria (from Marten's proposal)
ALL conditions must be satisfied simultaneously:
1. **Irregularly irregular rhythm** - highly variable RR intervals
2. **No distinct P-waves** - p_wave_crossings != 2 (baseline normal)
3. **Presence of f-waves** - fibrillatory baseline
4. **Narrow QRS complexes** - QRS < 120 ms combined with above

## ECG Interval Normal Ranges
- PR interval: 120-200 ms
- QRS duration: < 120 ms
- QT corrected: men ≤ 450 ms, women ≤ 470 ms
- Normal resting HR: 60-100 BPM

## Synthetic ECG Generation (ECGSynth_v2/)
Python pipeline to generate validated synthetic dataset (30 Normal, 30 AFib, 30 Noise) from the 9 source recordings using statistical simulation approach:

### Pipeline Steps:
1. `01_validate_recordings.py` - Verify raw ECG files
2. `02_preprocess.py` - Bandpass filter + baseline removal
3. `03_extract_features.py` - R-peaks, beat delineation, HRV metrics
4. `04_fit_statistics.py` - Fit distributions to beat parameters
5. `05_generate_normal.py` - Generate 30 Normal ECGs
6. `06_generate_afib.py` - Generate 30 AFib ECGs
7. `07_generate_noise.py` - Generate 30 Noise ECGs
8. `08_validate_generated.py` - Class-specific validation
9. `09_build_dataset.py` - Assemble final dataset

### Run: `python run_pipeline.py`

## Build System
- MPLAB X IDE project (Makefile-based)
- MCC (MPLAB Code Configurator) for peripheral setup
- Data Visualizer workspace for live plotting

## Key Data Structures
- `data_visualiser_data_t` in `src/DV_structure.h` - packed struct for serial visualization
- Fields include: raw sample, filtered stages, BPM, QRS width, P-wave metrics, classifier outputs

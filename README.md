# SonoFlow - Acoustic Uroflowmetry

A web application that reconstructs uroflowmetry curves from audio recordings of voiding.

## Features

- Upload 44.1kHz or 48kHz mono WAV recordings
- **Multi-episode voiding detection** (intermittent/straining patterns)
- Clinical parameters: Qmax, Qavg, Voiding Time, Flow Time
- Multiple Qmax methods (ICS-compliant sustained flow)
- Flow pattern classification (continuous/intermittent/straining)
- Downloadable graph with timestamp

## How It Works

The app uses acoustic signal processing to estimate urine flow rate from the sound of voiding:

1. **Audio preprocessing**: Band-pass filtering (250-4000 Hz) at native sample rate
2. **Energy extraction**: Short-time RMS energy computation
3. **Multi-episode detection**: Two-threshold approach (scanning + Otsu validation)
4. **Asymmetric edge refinement**: Changepoint for onset, tail-walk for offset
5. **Volume calibration**: Area under curve normalized to user-provided volume
6. **Smoothing**: Median filter + causal EMA for clinical-grade curves
7. **Qmax calculation**: 0.5s onset exclusion + 300ms sustained flow requirement
8. **ICS-compliant Qavg**: Uses flow time (excluding pauses), not voiding time

## ICS Compliance

| Metric | Definition |
|--------|------------|
| **Voiding Time** | Total duration from first to last flow (includes pauses) |
| **Flow Time** | Duration of actual flow only (excludes pauses) |
| **Qavg** | Volume ÷ Flow Time (per ICS guidelines) |

## Usage

1. Record your voiding audio (44.1kHz or 48kHz mono WAV recommended)
2. Upload the audio file
3. Enter the total voided volume (ml)
4. Click "Analyze Recording"
5. Download the generated graph

## Documentation

- [VOIDING_DETECTION.md](VOIDING_DETECTION.md) - Voiding detection algorithm details
- [QMAX_DETECTION.md](QMAX_DETECTION.md) - Qmax calculation methodology
- [API_DOCS.md](API_DOCS.md) - REST API documentation

## Disclaimer

This tool performs signal processing only. It does not provide medical diagnoses or clinical advice.

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

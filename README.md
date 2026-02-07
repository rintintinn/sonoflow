# SonoFlow - Acoustic Uroflowmetry

A web application that reconstructs uroflowmetry curves from audio recordings of voiding.

## Features

- Upload 44.1kHz or 48kHz mono WAV recordings
- Automatic voiding detection (Otsu + Changepoint adaptive)
- Clinical parameters: Qmax, Qavg, Voiding Time
- Multiple Qmax methods (ICS-compliant sustained flow)
- Downloadable graph with timestamp

## How It Works

The app uses acoustic signal processing to estimate urine flow rate from the sound of voiding:

1. **Audio preprocessing**: Band-pass filtering (250-4000 Hz) at native sample rate
2. **Energy extraction**: Short-time RMS energy computation
3. **Voiding detection**: Otsu adaptive thresholding + changepoint detection
4. **Volume calibration**: Area under curve normalized to user-provided volume
5. **Smoothing**: Median filter + causal EMA for clinical-grade curves
6. **Qmax calculation**: 0.5s onset exclusion + 300ms sustained flow requirement

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

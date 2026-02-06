# SonoFlow - Acoustic Uroflowmetry

A web application that reconstructs uroflowmetry curves from audio recordings of voiding.

## Features

- Upload WAV or MP3 audio recordings
- Automatic flow curve generation
- Clinical parameters: Qmax, Qavg, Voiding Time
- Downloadable graph with timestamp

## How It Works

The app uses acoustic signal processing to estimate urine flow rate from the sound of voiding:

1. **Audio preprocessing**: Band-pass filtering (250-4000 Hz)
2. **Energy extraction**: Short-time RMS energy computation
3. **Volume calibration**: Area under curve normalized to user-provided volume
4. **Smoothing**: Median filter + causal EMA for clinical-grade curves

## Usage

1. Record your voiding audio
2. Upload the audio file
3. Enter the total voided volume (ml)
4. Click "Analyze Recording"
5. Download the generated graph

## Disclaimer

This tool performs signal processing only. It does not provide medical diagnoses or clinical advice.

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

"""
FastAPI REST API for Acoustic Uroflowmetry

Provides programmatic access to the audio processing pipeline for third-party developers.
Runs alongside the Streamlit UI on a separate port (9996).
"""

import io
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np

from processor import AudioProcessor, validate_audio
from visualization import generate_clinical_graph
from version import __version__


app = FastAPI(
    title="SonoFlow API",
    description="Acoustic Uroflowmetry Processing API",
    version=__version__
)

# Enable CORS for third-party access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check and API information."""
    return {
        "service": "SonoFlow API",
        "version": __version__,
        "status": "healthy",
        "endpoints": {
            "/analyze": "POST - Returns JSON with parameters and base64 graph",
            "/analyze/download": "POST - Returns PNG graph directly"
        }
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Audio file (WAV or MP3)"),
    volume_ml: float = Form(..., description="Total voided volume in ml")
):
    """
    Analyze audio recording and return uroflowmetry parameters with graph.
    
    Returns JSON with:
    - success: boolean
    - parameters: dict with qmax_peak, qmax_smooth, qmax_icc_sliding, qmax_icc_consecutive, qavg, voiding_time, volume_ml
    - graph_base64: PNG image encoded as base64 string
    - timestamp: Analysis timestamp for reference
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load with librosa
        audio_data, sample_rate = librosa.load(audio_buffer, sr=None)
        duration = len(audio_data) / sample_rate
        
        # Validate input
        error = validate_audio(duration, sample_rate, volume_ml)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Process audio
        processor = AudioProcessor()
        result = processor.process(audio_data, sample_rate, volume_ml)
        
        # Generate graph
        graph_bytes, timestamp_str = generate_clinical_graph(result)
        graph_base64 = base64.b64encode(graph_bytes).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "parameters": {
                "qmax_peak": round(result.qmax, 1),
                "qmax_smooth": round(result.qmax_smoothed, 1),
                "qmax_icc_sliding": round(result.qmax_icc_sliding, 1),
                "qmax_icc_consecutive": round(result.qmax_icc_consecutive, 1),
                "qavg": round(result.qavg, 1),
                "voiding_time": round(result.voiding_time, 1),
                "volume_ml": result.volume_ml
            },
            "graph_base64": graph_base64,
            "timestamp": timestamp_str
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/analyze/download")
async def analyze_download(
    file: UploadFile = File(..., description="Audio file (WAV or MP3)"),
    volume_ml: float = Form(..., description="Total voided volume in ml")
):
    """
    Analyze audio recording and return the graph as a downloadable PNG.
    
    Returns: PNG image file
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load with librosa
        audio_data, sample_rate = librosa.load(audio_buffer, sr=None)
        duration = len(audio_data) / sample_rate
        
        # Validate input
        error = validate_audio(duration, sample_rate, volume_ml)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Process audio
        processor = AudioProcessor()
        result = processor.process(audio_data, sample_rate, volume_ml)
        
        # Generate graph
        graph_bytes, timestamp_str = generate_clinical_graph(result)
        
        return Response(
            content=graph_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="uroflowmetry_{timestamp_str}.png"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9996)

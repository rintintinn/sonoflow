# SonoFlow REST API Documentation

The SonoFlow project provides a FastAPI-based REST API for third-party developers to access the acoustic uroflowmetry processing pipeline programmatically.

## Base URL

When running locally:
`http://localhost:9996`

## Authentication

Currently, the API is open and does not require authentication.

---

## Endpoints

### 1. Health Check

Verifies that the API service is running.

- **URL**: `/`
- **Method**: `GET`
- **Response**: JSON with service status and available endpoints.

#### Example

```bash
curl http://localhost:9996/
```

---

### 2. Analyze Recording (Get Parameters & Graph)

Upload an audio recording and receive clinical parameters (including all 4 Qmax values) and a base64-encoded graph image.

- **URL**: `/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The audio file (WAV or MP3)
  - `volume_ml`: Total voided volume in ml (float)

#### Response

```json
{
  "success": true,
  "parameters": {
    "qmax_peak": 24.8,
    "qmax_smooth": 18.1,
    "qmax_icc_sliding": 22.1,
    "qmax_icc_consecutive": 18.8,
    "qavg": 7.4,
    "voiding_time": 26.9,
    "volume_ml": 200.0
  },
  "graph_base64": "iVBORw0KGgoAAAANSUhEUgAABcwAAAQJCAYAAAD7H0ukAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90...",
  "timestamp": "20240207_160828"
}
```

#### Curl Example

```bash
curl -X POST http://localhost:9996/analyze \
  -F "file=@/path/to/recording.wav" \
  -F "volume_ml=200"
```

---

### 3. Download Graph (Direct Image)

Upload an audio recording and directly download the generated clinical graph as a PNG image.

- **URL**: `/analyze/download`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The audio file (WAV or MP3)
  - `volume_ml`: Total voided volume in ml (float)

#### Response

- Binary PNG image file.

#### Curl Example

```bash
curl -X POST http://localhost:9996/analyze/download \
  -F "file=@/path/to/recording.wav" \
  -F "volume_ml=200" \
  -o uroflow_graph.png
```

---

## Error Handling

The API returns standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (e.g., audio too short, sample rate too low)
- `422`: Validation Error (missing parameters)
- `500`: Internal Server Error

## Running the API

To start the API server locally:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn python-multipart

# Run server on port 9996
uvicorn api:app --host 0.0.0.0 --port 9996 --reload
```

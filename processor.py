"""
Acoustic Uroflowmetry Audio Processor
8-step pipeline for converting audio to flow curve
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import librosa
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ProcessingResult:
    """Result of audio processing pipeline"""
    time: np.ndarray          # Time axis (seconds)
    flow_rate: np.ndarray     # Flow rate Q(t) in ml/s
    qmax: float               # Maximum flow rate from minimal smoothing (ml/s)
    qmax_smoothed: float      # Maximum flow rate from full smoothing (ml/s)
    qavg: float               # Average flow rate (ml/s)
    voiding_time: float       # Total voiding time (seconds)
    volume_ml: float          # User-provided volume (ml)


class AudioProcessor:
    """
    8-step audio processing pipeline for acoustic uroflowmetry.
    Converts audio recording to flow curve using acoustic analysis.
    """
    
    # Processing parameters
    TARGET_SR = 16000          # Target sample rate (Hz)
    FRAME_LENGTH_MS = 50       # Frame length (ms)
    FRAME_OVERLAP = 0.5        # 50% overlap
    LOWCUT = 250               # High-pass cutoff (Hz) - suppresses environmental noise
    HIGHCUT = 4000             # Band-pass high cutoff (Hz)
    NOISE_PERCENTILE = 10      # Percentile for noise floor estimation
    MEDIAN_FILTER_SIZE = 3     # Short median filter for transient spike suppression
    EMA_ALPHA = 0.15           # EMA smoothing factor (lower = smoother)
    
    def __init__(self):
        pass
    
    def process(self, audio_data: np.ndarray, sample_rate: int, volume_ml: float) -> ProcessingResult:
        """
        Main processing pipeline.
        
        Args:
            audio_data: Raw audio samples
            sample_rate: Original sample rate
            volume_ml: User-specified voided volume in ml
            
        Returns:
            ProcessingResult with flow curve and parameters
        """
        # Step 1: Input normalization
        audio, sr = self._normalize_input(audio_data, sample_rate)
        
        # Step 2: Band-pass filtering
        filtered = self._bandpass_filter(audio, sr)
        
        # Step 3: Frame-based energy extraction
        time_axis, energy = self._extract_energy(filtered, sr)
        
        # Step 4: Noise floor calibration
        noise_floor = self._calibrate_noise_floor(energy)
        
        # Step 5: Voiding segment detection
        voiding_mask, voiding_time = self._detect_voiding(energy, noise_floor, time_axis)
        
        # Step 6: Flow proxy construction
        flow_proxy = self._construct_flow_proxy(energy, noise_floor)
        
        # Step 7: Volume normalization
        flow_rate = self._normalize_volume(flow_proxy, volume_ml, time_axis)
        
        # Step 8: Smoothing (with separate Qmax calculation)
        flow_rate_minimal, flow_rate_smooth = self._smooth_signal(flow_rate)
        
        # Calculate parameters - both Qmax values for validation
        qmax = float(np.max(flow_rate_minimal))        # From minimal smoothing
        qmax_smoothed = float(np.max(flow_rate_smooth)) # From full smoothing
        qavg = volume_ml / voiding_time if voiding_time > 0 else 0.0
        
        return ProcessingResult(
            time=time_axis,
            flow_rate=flow_rate_smooth,
            qmax=qmax,
            qmax_smoothed=qmax_smoothed,
            qavg=qavg,
            voiding_time=voiding_time,
            volume_ml=volume_ml
        )
    
    def _normalize_input(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Step 1: Convert to mono, resample to 16kHz"""
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to target sample rate
        if sr != self.TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.TARGET_SR)
        
        # Normalize amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        return audio, self.TARGET_SR
    
    def _bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Step 2: Apply band-pass filtering using cascaded high-pass and low-pass filters.
        
        Uses a 3rd-order Butterworth high-pass filter at 250 Hz to suppress 
        low-frequency environmental noise and stabilize noise floor estimation,
        followed by a 3rd-order low-pass filter at 4000 Hz.
        """
        nyquist = sr / 2
        
        # High-pass filter: 3rd-order Butterworth at 250 Hz
        # (midpoint of 200-300 Hz range for environmental noise suppression)
        hp_cutoff = self.LOWCUT / nyquist
        b_hp, a_hp = signal.butter(3, hp_cutoff, btype='high')
        
        # Low-pass filter: 3rd-order Butterworth at 4000 Hz
        lp_cutoff = self.HIGHCUT / nyquist
        b_lp, a_lp = signal.butter(3, lp_cutoff, btype='low')
        
        # Apply filters in cascade (high-pass first, then low-pass)
        filtered = signal.filtfilt(b_hp, a_hp, audio)
        filtered = signal.filtfilt(b_lp, a_lp, filtered)
        
        return filtered
    
    def _extract_energy(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Step 3: Compute RMS energy per frame with light smoothing"""
        frame_length = int(self.FRAME_LENGTH_MS * sr / 1000)
        hop_length = int(frame_length * (1 - self.FRAME_OVERLAP))
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Apply light smoothing to RMS to reduce high-frequency fluctuations
        # This helps create smoother flow curves similar to medical devices
        if len(rms) >= 5:
            from scipy.ndimage import uniform_filter1d
            rms = uniform_filter1d(rms, size=5, mode='nearest')
        
        # Create time axis
        time_axis = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        return time_axis, rms
    
    def _calibrate_noise_floor(self, energy: np.ndarray) -> float:
        """Step 4: Estimate noise floor from lowest percentile"""
        noise_floor = np.percentile(energy, self.NOISE_PERCENTILE)
        return noise_floor
    
    def _detect_voiding(self, energy: np.ndarray, noise_floor: float, 
                        time_axis: np.ndarray) -> Tuple[np.ndarray, float]:
        """Step 5: Detect voiding segments and calculate total time"""
        # Threshold slightly above noise floor
        threshold = noise_floor * 1.5
        
        # Find frames above threshold
        voiding_mask = energy > threshold
        
        if not np.any(voiding_mask):
            return voiding_mask, 0.0
        
        # Find first and last voiding frame
        voiding_indices = np.where(voiding_mask)[0]
        first_idx = voiding_indices[0]
        last_idx = voiding_indices[-1]
        
        # Calculate voiding time
        voiding_time = time_axis[last_idx] - time_axis[first_idx]
        
        return voiding_mask, float(voiding_time)
    
    def _construct_flow_proxy(self, energy: np.ndarray, noise_floor: float) -> np.ndarray:
        """Step 6: F_proxy(t) = max(E(t) - noise_floor, 0)"""
        flow_proxy = np.maximum(energy - noise_floor, 0)
        return flow_proxy
    
    def _normalize_volume(self, flow_proxy: np.ndarray, volume_ml: float,
                          time_axis: np.ndarray) -> np.ndarray:
        """Step 7: Scale so total area equals voided volume"""
        # Calculate time step
        dt = np.mean(np.diff(time_axis)) if len(time_axis) > 1 else 1.0
        
        # Calculate area under proxy curve
        area = np.trapz(flow_proxy, dx=dt)
        
        if area < 1e-10:
            return flow_proxy
        
        # Scale factor
        k = volume_ml / area
        
        # Normalized flow rate
        flow_rate = k * flow_proxy
        
        return flow_rate
    
    def _smooth_signal(self, flow_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 8: Apply median + causal EMA smoothing.
        
        Uses a short median filter to suppress transient acoustic spikes,
        followed by a causal exponential moving average (EMA) to produce
        a physiologically plausible flow envelope.
        
        Returns:
            Tuple of (minimally_smoothed, fully_smoothed) for Qmax calculation
        """
        if len(flow_rate) < 3:
            return flow_rate, flow_rate
        
        # Stage 1: Short median filter to suppress transient spikes
        from scipy.ndimage import median_filter
        minimal_smooth = median_filter(flow_rate, size=self.MEDIAN_FILTER_SIZE, mode='nearest')
        minimal_smooth = np.maximum(minimal_smooth, 0)
        
        # Stage 2: Causal exponential moving average for physiological flow envelope
        ema_smoothed = self._causal_ema(minimal_smooth, self.EMA_ALPHA)
        
        # Ensure non-negative
        ema_smoothed = np.maximum(ema_smoothed, 0)
        
        return minimal_smooth, ema_smoothed
    
    def _causal_ema(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply causal (forward-only) exponential moving average.
        
        EMA formula: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
        
        Args:
            data: Input signal
            alpha: Smoothing factor (0 < alpha <= 1). Lower = smoother.
            
        Returns:
            Smoothed signal
        """
        result = np.zeros_like(data)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result


def validate_audio(duration: float, sample_rate: int, volume_ml: float) -> str | None:
    """Validate audio parameters. Returns error message or None if valid."""
    if duration < 3.0:
        return f"Audio too short: {duration:.1f}s (minimum 3s required)"
    if sample_rate < 8000:
        return f"Sample rate too low: {sample_rate}Hz (minimum 8kHz required)"
    if volume_ml <= 0:
        return "Volume must be greater than 0"
    return None

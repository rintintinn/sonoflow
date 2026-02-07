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
    qmax_icc_sliding: float   # ICC: max of sliding window average (~200ms)
    qmax_icc_consecutive: float  # ICC: sustained max (consecutive frames above threshold)
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
    ICC_WINDOW_FRAMES = 8      # ~200ms at 25ms hop (for ICC Qmax calculation)
    ICC_THRESHOLD_RATIO = 0.95 # Threshold for consecutive frames method
    ONSET_THRESHOLD_MULT = 2.0 # Multiplier for onset detection (above noise floor)
    MIN_VOIDING_FRAMES = 20    # Minimum frames to consider as voiding (~500ms)
    
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
        
        # Step 5: Voiding segment detection (improved: find actual flow start/end)
        start_idx, end_idx, voiding_time = self._detect_voiding_segment(energy, noise_floor, time_axis)
        
        # Step 6: Flow proxy construction
        flow_proxy = self._construct_flow_proxy(energy, noise_floor)
        
        # Trim to voiding segment and shift time to start at 0
        flow_proxy_trimmed = flow_proxy[start_idx:end_idx+1]
        time_axis_trimmed = time_axis[start_idx:end_idx+1] - time_axis[start_idx]
        
        # Step 7: Volume normalization (on trimmed data)
        flow_rate = self._normalize_volume(flow_proxy_trimmed, volume_ml, time_axis_trimmed)
        
        # Step 8: Smoothing (with separate Qmax calculation)
        flow_rate_minimal, flow_rate_smooth = self._smooth_signal(flow_rate)
        
        # Calculate parameters - multiple Qmax values for validation
        qmax = float(np.max(flow_rate_minimal))        # From minimal smoothing
        qmax_smoothed = float(np.max(flow_rate_smooth)) # From full smoothing
        
        # ICC-compliant Qmax calculations (sustained >=200ms)
        qmax_icc_sliding = self._calc_qmax_icc_sliding(flow_rate_minimal)
        qmax_icc_consecutive = self._calc_qmax_icc_consecutive(flow_rate_minimal)
        
        qavg = volume_ml / voiding_time if voiding_time > 0 else 0.0
        
        return ProcessingResult(
            time=time_axis_trimmed,
            flow_rate=flow_rate_smooth,
            qmax=qmax,
            qmax_smoothed=qmax_smoothed,
            qmax_icc_sliding=qmax_icc_sliding,
            qmax_icc_consecutive=qmax_icc_consecutive,
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
    
    def _detect_voiding_segment(self, energy: np.ndarray, noise_floor: float, 
                                time_axis: np.ndarray) -> Tuple[int, int, float]:
        """
        Step 5: Detect the actual voiding segment.
        
        Uses a higher threshold (3x noise) to distinguish actual voiding from 
        background noise, and identifies the main contiguous voiding region
        while filtering out isolated post-void spikes.
        
        Returns:
            start_idx: Index of voiding start
            end_idx: Index of voiding end
            voiding_time: Duration in seconds
        """
        # Use 3x noise floor threshold - empirically determined to distinguish
        # voiding sound from background noise in bathroom environment
        flow_threshold = noise_floor * 3.0
        
        # Find all frames above threshold
        above_threshold = energy > flow_threshold
        
        if not np.any(above_threshold):
            # Fallback to entire recording if no clear flow detected
            return 0, len(energy) - 1, float(time_axis[-1] - time_axis[0])
        
        # Find all indices above threshold
        threshold_indices = np.where(above_threshold)[0]
        first_idx = threshold_indices[0]
        
        # Find peak energy for relative end detection
        peak_energy = np.max(energy)
        peak_idx = np.argmax(energy)
        
        # Find end of voiding: first point after peak where energy drops to <10% of peak
        # and stays low for at least 0.5 seconds (20 frames)
        end_threshold = peak_energy * 0.10  # 10% of peak
        SUSTAINED_LOW_FRAMES = 20  # ~0.5 seconds
        
        true_end_idx = threshold_indices[-1]  # Default to last high frame
        
        for i in range(peak_idx, len(energy)):
            if energy[i] < end_threshold:
                # Check if it stays low for SUSTAINED_LOW_FRAMES
                low_count = 0
                for j in range(i, min(i + SUSTAINED_LOW_FRAMES, len(energy))):
                    if energy[j] < end_threshold:
                        low_count += 1
                    else:
                        break
                
                if low_count >= SUSTAINED_LOW_FRAMES:
                    true_end_idx = i
                    break
        
        last_idx = true_end_idx
        
        # Refine start: walk back to find where upslope begins
        start_idx = first_idx
        for i in range(first_idx, max(first_idx - 40, 0), -1):
            if energy[i] < noise_floor * 1.5:
                start_idx = i + 1
                break
        else:
            start_idx = max(first_idx - 40, 0)
        
        # Refine end: include any trailing flow above noise
        end_idx = last_idx
        for i in range(last_idx, min(last_idx + 20, len(energy) - 1)):
            if energy[i] < noise_floor * 1.5:
                end_idx = i
                break
        else:
            end_idx = min(last_idx + 20, len(energy) - 1)
        
        # Ensure valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(energy) - 1, end_idx)
        
        voiding_time = time_axis[end_idx] - time_axis[start_idx]
        
        return int(start_idx), int(end_idx), float(voiding_time)
    
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
    
    def _calc_qmax_icc_sliding(self, flow_rate: np.ndarray) -> float:
        """
        ICC Qmax Method 1: Sliding window average.
        
        Calculates Qmax as the maximum of sliding window averages,
        where window size corresponds to ~200ms (8 frames at 25ms hop).
        This ensures Qmax represents a sustained flow, not a transient spike.
        """
        if len(flow_rate) < self.ICC_WINDOW_FRAMES:
            return float(np.max(flow_rate))
        
        # Compute sliding window average
        kernel = np.ones(self.ICC_WINDOW_FRAMES) / self.ICC_WINDOW_FRAMES
        sliding_avg = np.convolve(flow_rate, kernel, mode='valid')
        
        return float(np.max(sliding_avg))
    
    def _calc_qmax_icc_consecutive(self, flow_rate: np.ndarray) -> float:
        """
        ICC Qmax Method 2: Consecutive frames above threshold.
        
        Finds the highest flow value that is sustained for at least
        ICC_WINDOW_FRAMES consecutive frames (≥200ms).
        Uses binary search to find the maximum sustained threshold.
        """
        if len(flow_rate) < self.ICC_WINDOW_FRAMES:
            return float(np.max(flow_rate))
        
        peak = np.max(flow_rate)
        min_val = 0.0
        max_val = peak
        
        # Binary search for highest sustained threshold
        for _ in range(20):  # 20 iterations gives good precision
            mid = (min_val + max_val) / 2
            
            # Check if there are N consecutive frames above mid
            above_threshold = flow_rate >= mid
            
            # Find longest run of True values
            max_run = 0
            current_run = 0
            for val in above_threshold:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            
            if max_run >= self.ICC_WINDOW_FRAMES:
                min_val = mid  # This threshold is achievable
            else:
                max_val = mid  # This threshold is too high
        
        return float(min_val)


def validate_audio(duration: float, sample_rate: int, volume_ml: float) -> str | None:
    """Validate audio parameters. Returns error message or None if valid."""
    if duration < 3.0:
        return f"Audio too short: {duration:.1f}s (minimum 3s required)"
    if sample_rate < 8000:
        return f"Sample rate too low: {sample_rate}Hz (minimum 8kHz required)"
    if volume_ml <= 0:
        return "Volume must be greater than 0"
    return None

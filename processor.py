"""
Acoustic Uroflowmetry Audio Processor
8-step pipeline for converting audio to flow curve
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import librosa
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class ProcessingResult:
    """Result of audio processing pipeline"""
    time: np.ndarray          # Time axis (seconds)
    flow_rate: np.ndarray     # Flow rate Q(t) in ml/s
    qmax: float               # Maximum flow rate from minimal smoothing (ml/s)
    qmax_smoothed: float      # Maximum flow rate from full smoothing (ml/s)
    qmax_ics_sliding: float   # ICS: max of sliding window average (~300ms)
    qmax_ics_consecutive: float  # ICS: sustained max (consecutive frames above threshold)
    qavg: float               # Average flow rate (ml/s)
    voiding_time: float       # Total voiding time including pauses (ICS "voiding time")
    volume_ml: float          # User-provided volume (ml)
    
    # Multi-episode detection (ICS-compliant)
    num_episodes: int = 1                   # Number of flow episodes detected
    flow_pattern: str = "continuous"        # "continuous", "intermittent", "straining"
    flow_time: Optional[float] = None       # Actual flow time excluding pauses (ICS "flow time")
    
    # Audio quality indicators
    sample_rate: int = 0                    # Original sample rate
    snr_db: float = 0.0                     # Signal-to-noise ratio in dB
    quality_warning: Optional[str] = None   # Warning message if quality issue detected
    
    # Alternative detection results (optional)
    alt_start_time: Optional[float] = None
    alt_end_time: Optional[float] = None
    alt_voiding_time: Optional[float] = None
    alt_otsu_threshold: Optional[float] = None
    
    # Slope-stabilized Qmax (debug only)
    qmax_slope_stabilized: Optional[float] = None
    slope_threshold: Optional[float] = None
    
    # Debug data (optional, only populated when debug=True)
    debug_data: Optional[Dict[str, Any]] = field(default=None)



class AudioProcessor:
    """
    8-step audio processing pipeline for acoustic uroflowmetry.
    Converts audio recording to flow curve using acoustic analysis.
    """
    
    # Processing parameters (native sample rate - no resampling)
    FRAME_LENGTH_MS = 50       # Frame length (ms)
    FRAME_OVERLAP = 0.5        # 50% overlap
    LOWCUT = 250               # High-pass cutoff (Hz) - suppresses environmental noise
    HIGHCUT = 4000             # Band-pass high cutoff (Hz)
    NOISE_PERCENTILE = 10      # Percentile for noise floor estimation
    MEDIAN_FILTER_SIZE = 3     # Short median filter for transient spike suppression
    EMA_ALPHA = 0.15           # EMA smoothing factor (lower = smoother)
    ICS_WINDOW_FRAMES = 12     # ~300ms at 25ms hop (for ICS Qmax calculation)
    ICS_THRESHOLD_RATIO = 0.95 # Threshold for consecutive frames method
    ONSET_THRESHOLD_MULT = 2.0 # Multiplier for onset detection (above noise floor)
    MIN_VOIDING_FRAMES = 20    # Minimum frames to consider as voiding (~500ms)
    QMAX_ONSET_EXCLUSION_SEC = 0.5  # Exclude first 0.5s from Qmax (transient splash)
    
    # Slope stabilization parameters (for debug Qmax)
    SLOPE_PERCENTILE = 90      # Percentile for adaptive slope threshold
    SLOPE_STABLE_FRAMES = 8    # ~200ms for slope stabilization requirement
    QMAX_SUSTAINED_FRAMES = 12 # ~300ms for Qmax sustained requirement
    
    def __init__(self):
        pass
    
    def process(self, audio_data: np.ndarray, sample_rate: int, volume_ml: float, 
                debug: bool = False) -> ProcessingResult:
        """
        Main processing pipeline.
        
        Args:
            audio_data: Raw audio samples
            sample_rate: Original sample rate
            volume_ml: User-specified voided volume in ml
            debug: If True, also run alternative detection and store debug data
            
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
        
        # Step 4b: Quality checks - sample rate and SNR
        quality_warnings = []
        
        # Check sample rate (expect 44.1kHz or 48kHz)
        if sample_rate not in [44100, 48000]:
            quality_warnings.append(f"Non-standard sample rate: {sample_rate}Hz (expected 44100 or 48000)")
        
        # Step 5: Voiding segment detection using Multi-Episode detection
        # Uses Otsu adaptive threshold + changepoint detection (consistent with alternative_detection.py)
        from multi_episode_detection import detect_voiding_multiepisode
        
        multi_result = detect_voiding_multiepisode(
            energy=energy,
            time_axis=time_axis,
        )
        start_idx = multi_result.voiding_start_idx
        end_idx = multi_result.voiding_end_idx
        voiding_time = multi_result.voiding_time
        flow_time = multi_result.flow_time
        num_episodes = multi_result.num_episodes
        flow_pattern = multi_result.pattern
        otsu_threshold_value = multi_result.otsu_thresh
        
        # Step 5b: Compute SNR (Signal-to-Noise Ratio)
        # SNR = 10 * log10(voiding_energy / noise_floor)
        voiding_energy = energy[start_idx:end_idx+1]
        if len(voiding_energy) > 0 and noise_floor > 0:
            mean_voiding_energy = float(np.mean(voiding_energy))
            snr_db = 10 * np.log10(mean_voiding_energy / noise_floor + 1e-10)
        else:
            snr_db = 0.0
        
        # Flag low SNR recordings as unreliable
        if snr_db < 10.0:
            quality_warnings.append(f"Low SNR: {snr_db:.1f} dB (<10 dB may be unreliable)")
        
        # Combine warnings
        quality_warning = "; ".join(quality_warnings) if quality_warnings else None
        
        # Step 6: Flow proxy construction
        flow_proxy = self._construct_flow_proxy(energy, noise_floor)
        
        # Trim to voiding segment and shift time to start at 0
        flow_proxy_trimmed = flow_proxy[start_idx:end_idx+1]
        time_axis_trimmed = time_axis[start_idx:end_idx+1] - time_axis[start_idx]
        
        # Step 7: Volume normalization (on trimmed data)
        flow_rate = self._normalize_volume(flow_proxy_trimmed, volume_ml, time_axis_trimmed)
        
        # Step 8: Smoothing (with separate Qmax calculation)
        flow_rate_minimal, flow_rate_smooth = self._smooth_signal(flow_rate)
        
        # Apply onset exclusion window for Qmax calculation
        # Exclude first 0.5s to avoid transient "splash" spikes at voiding onset
        hop_length_sec = (self.FRAME_LENGTH_MS / 1000) * (1 - self.FRAME_OVERLAP)
        exclusion_frames = int(self.QMAX_ONSET_EXCLUSION_SEC / hop_length_sec)
        
        # Create masked arrays for Qmax calculation (excluding onset window)
        if len(flow_rate_minimal) > exclusion_frames:
            flow_for_qmax_minimal = flow_rate_minimal[exclusion_frames:]
            flow_for_qmax_smooth = flow_rate_smooth[exclusion_frames:]
        else:
            # Recording too short, use full signal
            flow_for_qmax_minimal = flow_rate_minimal
            flow_for_qmax_smooth = flow_rate_smooth
        
        # Calculate parameters - multiple Qmax values for validation
        # All Qmax measurements now exclude the onset transient window
        qmax = float(np.max(flow_for_qmax_minimal))        # From minimal smoothing
        qmax_smoothed = float(np.max(flow_for_qmax_smooth)) # From full smoothing
        
        # ICS-compliant Qmax calculations (sustained >=300ms)
        qmax_ics_sliding = self._calc_qmax_ics_sliding(flow_for_qmax_minimal)
        qmax_ics_consecutive = self._calc_qmax_ics_consecutive(flow_for_qmax_minimal)
        
        # ICS-compliant Qavg: uses FLOW TIME (excluding pauses), not voiding time
        qavg = volume_ml / flow_time if flow_time and flow_time > 0 else 0.0
        
        # Initialize optional fields
        alt_start_time = None
        alt_end_time = None
        alt_voiding_time = None
        alt_otsu_threshold = otsu_threshold_value  # Store Otsu threshold from multi-episode detection
        qmax_slope_stabilized = None
        slope_threshold = None
        debug_data = None
        
        # Run legacy detection and slope-stabilized Qmax if debug mode is enabled
        if debug:
            # Run legacy fixed-threshold detection for comparison
            legacy_start_idx, legacy_end_idx, legacy_voiding_time = self._detect_voiding_segment(
                energy, noise_floor, time_axis
            )
            alt_start_time = float(time_axis[legacy_start_idx])
            alt_end_time = float(time_axis[legacy_end_idx])
            alt_voiding_time = legacy_voiding_time
            
            # Compute slope-stabilized Qmax (on full trimmed flow, not exclusion-masked)
            qmax_slope_stabilized, slope_threshold, stable_mask = self._calc_qmax_slope_stabilized(
                flow_rate_minimal, hop_length_sec
            )
            
            # Store debug data for visualization
            # Note: "fixed" = Otsu+changepoint (now default), "alt" = legacy fixed-threshold
            debug_data = {
                'filtered_audio': filtered,
                'sample_rate': sr,
                'time_axis_full': time_axis,
                'energy': energy,
                'noise_floor': noise_floor,
                'fixed_start_idx': start_idx,  # Otsu+changepoint (default)
                'fixed_end_idx': end_idx,
                'alt_start_idx': legacy_start_idx,  # Legacy fixed-threshold
                'alt_end_idx': legacy_end_idx,
                # Slope-stabilized Qmax debug data
                'flow_rate_minimal': flow_rate_minimal,
                'flow_rate_smooth': flow_rate_smooth,
                'time_axis_trimmed': time_axis_trimmed,
                'stable_mask': stable_mask,
                'slope_threshold': slope_threshold,
                'qmax_slope_stabilized': qmax_slope_stabilized,
            }
        
        return ProcessingResult(
            time=time_axis_trimmed,
            flow_rate=flow_rate_smooth,
            qmax=qmax,
            qmax_smoothed=qmax_smoothed,
            qmax_ics_sliding=qmax_ics_sliding,
            qmax_ics_consecutive=qmax_ics_consecutive,
            qavg=qavg,
            voiding_time=voiding_time,
            volume_ml=volume_ml,
            num_episodes=num_episodes,
            flow_pattern=flow_pattern,
            flow_time=flow_time,
            sample_rate=sample_rate,
            snr_db=snr_db,
            quality_warning=quality_warning,
            alt_start_time=alt_start_time,
            alt_end_time=alt_end_time,
            alt_voiding_time=alt_voiding_time,
            alt_otsu_threshold=alt_otsu_threshold,
            qmax_slope_stabilized=qmax_slope_stabilized,
            slope_threshold=slope_threshold,
            debug_data=debug_data
        )
    
    def _normalize_input(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Step 1: Convert to mono, normalize amplitude (no resampling).
        
        Expects 44.1kHz or 48kHz mono uncompressed WAV.
        Bandpass filter is applied at native sample rate.
        """
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize amplitude (no resampling - use native rate)
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        return audio, sr
    
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
    
    def _calc_qmax_ics_sliding(self, flow_rate: np.ndarray) -> float:
        """
        ICS Qmax Method 1: Sliding window average.
        
        Calculates Qmax as the maximum of sliding window averages,
        where window size corresponds to ~300ms (12 frames at 25ms hop).
        This ensures Qmax represents a sustained flow, not a transient spike.
        """
        if len(flow_rate) < self.ICS_WINDOW_FRAMES:
            return float(np.max(flow_rate))
        
        # Compute sliding window average
        kernel = np.ones(self.ICS_WINDOW_FRAMES) / self.ICS_WINDOW_FRAMES
        sliding_avg = np.convolve(flow_rate, kernel, mode='valid')
        
        return float(np.max(sliding_avg))
    
    def _calc_qmax_ics_consecutive(self, flow_rate: np.ndarray) -> float:
        """
        ICS Qmax Method 2: Consecutive frames above threshold.
        
        Finds the highest flow value that is sustained for at least
        ICS_WINDOW_FRAMES consecutive frames (≥300ms).
        Uses binary search to find the maximum sustained threshold.
        """
        if len(flow_rate) < self.ICS_WINDOW_FRAMES:
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
            
            if max_run >= self.ICS_WINDOW_FRAMES:
                min_val = mid  # This threshold is achievable
            else:
                max_val = mid  # This threshold is too high
        
        return float(min_val)
    
    def _calc_qmax_slope_stabilized(self, flow_rate: np.ndarray, 
                                      hop_length_sec: float) -> Tuple[float, float, np.ndarray]:
        """
        Slope-stabilized Qmax: Only considers stable flow regions.
        
        Algorithm:
        1. Compute first-order difference dQ/dt from flow signal
        2. Use adaptive threshold (90th percentile of |dQ/dt|)
        3. Mark regions where |dQ/dt| ≤ threshold for ≥200ms as "stable"
        4. Within stable regions, find max flow sustained for ≥300ms
        
        Returns:
            Tuple of (qmax_value, slope_threshold, stable_mask)
            - qmax_value: Maximum sustained flow in stable regions
            - slope_threshold: Adaptive threshold used
            - stable_mask: Boolean array marking stable time points
        """
        if len(flow_rate) < self.SLOPE_STABLE_FRAMES + self.QMAX_SUSTAINED_FRAMES:
            # Signal too short, return simple max
            return float(np.max(flow_rate)), 0.0, np.ones(len(flow_rate), dtype=bool)
        
        # Step 1: Compute first-order difference (dQ/dt)
        # dQ_dt[t] = (Q[t] - Q[t-1]) / dt
        dt = hop_length_sec
        dQ_dt = np.diff(flow_rate) / dt
        
        # Pad to match original length (first point has no derivative)
        dQ_dt = np.concatenate([[0], dQ_dt])
        
        # Step 2: Compute absolute slope
        abs_dQ_dt = np.abs(dQ_dt)
        
        # Step 3: Adaptive threshold at 90th percentile
        # EXCLUDE first 300ms from threshold calibration to prevent onset turbulence inflation
        slope_exclusion_frames = int(0.3 / hop_length_sec)  # 300ms = ~12 frames at 25ms hop
        if len(abs_dQ_dt) > slope_exclusion_frames:
            abs_dQ_dt_for_threshold = abs_dQ_dt[slope_exclusion_frames:]
        else:
            abs_dQ_dt_for_threshold = abs_dQ_dt
        
        slope_threshold = float(np.percentile(abs_dQ_dt_for_threshold, self.SLOPE_PERCENTILE))
        
        # Avoid zero threshold
        if slope_threshold < 1e-6:
            slope_threshold = 1e-6
        
        # Step 4: Mark points where slope is below threshold
        below_threshold = abs_dQ_dt <= slope_threshold
        
        # Step 5: Find regions where slope is stable for ≥200ms (SLOPE_STABLE_FRAMES)
        # A point is "stable" if it's part of a run of ≥N consecutive low-slope points
        stable_mask = np.zeros(len(flow_rate), dtype=bool)
        
        current_run_start = 0
        current_run_length = 0
        
        for i in range(len(below_threshold)):
            if below_threshold[i]:
                if current_run_length == 0:
                    current_run_start = i
                current_run_length += 1
            else:
                # End of run - mark if long enough
                if current_run_length >= self.SLOPE_STABLE_FRAMES:
                    stable_mask[current_run_start:current_run_start + current_run_length] = True
                current_run_length = 0
        
        # Handle final run
        if current_run_length >= self.SLOPE_STABLE_FRAMES:
            stable_mask[current_run_start:current_run_start + current_run_length] = True
        
        # Step 6: Within stable regions, find max sustained Qmax (≥300ms)
        # Apply stable mask to flow and find sustained max
        if not np.any(stable_mask):
            # No stable regions found, fall back to simple max
            return float(np.max(flow_rate)), slope_threshold, stable_mask
        
        # Create a masked flow (NaN for unstable regions)
        flow_stable = np.where(stable_mask, flow_rate, np.nan)
        
        # Find sustained max using sliding window within stable regions
        if np.sum(stable_mask) < self.QMAX_SUSTAINED_FRAMES:
            # Not enough stable points, return max of stable points
            return float(np.nanmax(flow_stable)), slope_threshold, stable_mask
        
        # Compute sliding window average within stable regions
        qmax_stable = 0.0
        for i in range(len(flow_rate) - self.QMAX_SUSTAINED_FRAMES + 1):
            window = flow_stable[i:i + self.QMAX_SUSTAINED_FRAMES]
            if np.all(~np.isnan(window)):  # All points in window are stable
                window_avg = np.mean(window)
                qmax_stable = max(qmax_stable, window_avg)
        
        # If no full stable window found, use max of stable points
        if qmax_stable == 0.0:
            qmax_stable = float(np.nanmax(flow_stable))
        
        return qmax_stable, slope_threshold, stable_mask


def validate_audio(duration: float, sample_rate: int, volume_ml: float) -> str | None:
    """Validate audio parameters. Returns error message or None if valid."""
    if duration < 3.0:
        return f"Audio too short: {duration:.1f}s (minimum 3s required)"
    if sample_rate < 8000:
        return f"Sample rate too low: {sample_rate}Hz (minimum 8kHz required)"
    if volume_ml <= 0:
        return "Volume must be greater than 0"
    return None

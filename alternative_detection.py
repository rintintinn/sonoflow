"""
Alternative Detection Methods for Acoustic Uroflowmetry

Provides alternative onset/end detection using:
- Otsu adaptive thresholding
- Rolling-statistics changepoint detection

These methods run in parallel with the fixed-threshold approach
for validation and comparison.

Design: Modular architecture allows future integration of
library-based methods (e.g., ruptures PELT) as drop-in replacements.
"""

import numpy as np
from typing import Tuple, Protocol, Optional
from dataclasses import dataclass


# ============================================================================
# Changepoint Detector Protocol (for future extensibility)
# ============================================================================

class ChangePointDetector(Protocol):
    """Protocol for changepoint detection implementations.
    
    Allows swapping between rolling-stats and library-based (e.g., ruptures)
    implementations without changing the calling code.
    """
    
    def detect(self, signal: np.ndarray, min_size: int = 10) -> list[int]:
        """Detect changepoints in the signal.
        
        Args:
            signal: 1D signal to analyze
            min_size: Minimum segment size between changepoints
            
        Returns:
            List of changepoint indices (sorted)
        """
        ...


# ============================================================================
# Rolling Statistics Changepoint Detector (dependency-free)
# ============================================================================

class RollingStatsChangePointDetector:
    """
    Changepoint detection using rolling statistics.
    
    Compares short-term vs long-term mean to identify sustained changes.
    This approach is lightweight and requires no external dependencies.
    
    Algorithm:
    1. Compute short-term rolling mean (e.g., 5 frames)
    2. Compute long-term rolling mean (e.g., 20 frames)
    3. Calculate ratio: short_term / long_term
    4. Identify points where ratio exceeds threshold
    5. Filter by minimum duration constraint
    
    Why this works:
    - Voiding onset causes sustained increase in energy
    - Short-term mean rises faster than long-term baseline
    - Ratio >> 1 indicates onset, ratio << 1 indicates end
    """
    
    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        onset_ratio_threshold: float = 1.5,
        end_ratio_threshold: float = 0.7
    ):
        """
        Args:
            short_window: Frames for short-term mean
            long_window: Frames for long-term mean
            onset_ratio_threshold: Ratio threshold for onset detection
            end_ratio_threshold: Ratio threshold for end detection
        """
        self.short_window = short_window
        self.long_window = long_window
        self.onset_ratio_threshold = onset_ratio_threshold
        self.end_ratio_threshold = end_ratio_threshold
    
    def _rolling_mean(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean with edge handling."""
        kernel = np.ones(window) / window
        # Use 'same' mode and handle edges
        padded = np.pad(signal, (window // 2, window - window // 2 - 1), mode='edge')
        return np.convolve(padded, kernel, mode='valid')
    
    def detect(self, signal: np.ndarray, min_size: int = 10) -> list[int]:
        """
        Detect changepoints using rolling mean ratio.
        
        Returns indices where significant changes occur.
        """
        if len(signal) < self.long_window:
            return []
        
        short_mean = self._rolling_mean(signal, self.short_window)
        long_mean = self._rolling_mean(signal, self.long_window)
        
        # Avoid division by zero
        long_mean = np.maximum(long_mean, 1e-10)
        ratio = short_mean / long_mean
        
        # Find onset candidates (ratio rises above threshold)
        onset_candidates = np.where(
            (ratio[1:] >= self.onset_ratio_threshold) & 
            (ratio[:-1] < self.onset_ratio_threshold)
        )[0] + 1
        
        # Find end candidates (ratio falls below threshold)  
        end_candidates = np.where(
            (ratio[1:] <= self.end_ratio_threshold) & 
            (ratio[:-1] > self.end_ratio_threshold)
        )[0] + 1
        
        # Combine and sort
        changepoints = sorted(set(onset_candidates.tolist() + end_candidates.tolist()))
        
        # Filter by minimum segment size
        if len(changepoints) > 1:
            filtered = [changepoints[0]]
            for cp in changepoints[1:]:
                if cp - filtered[-1] >= min_size:
                    filtered.append(cp)
            changepoints = filtered
        
        return changepoints
    
    def detect_onset_end(
        self,
        signal: np.ndarray,
        min_duration_frames: int = 8
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Detect voiding onset and end using rolling statistics.
        
        Args:
            signal: Energy signal
            min_duration_frames: Minimum frames above/below threshold
            
        Returns:
            Tuple of (onset_idx, end_idx) or (None, None) if not found
        """
        if len(signal) < self.long_window:
            return None, None
        
        short_mean = self._rolling_mean(signal, self.short_window)
        long_mean = self._rolling_mean(signal, self.long_window)
        long_mean = np.maximum(long_mean, 1e-10)
        ratio = short_mean / long_mean
        
        # Find onset: first sustained period above threshold
        above_onset = ratio > self.onset_ratio_threshold
        onset_idx = None
        for i in range(len(above_onset) - min_duration_frames):
            if np.all(above_onset[i:i + min_duration_frames]):
                onset_idx = i
                break
        
        # Find end: last sustained period above threshold
        end_idx = None
        if onset_idx is not None:
            for i in range(len(above_onset) - 1, onset_idx + min_duration_frames, -1):
                if np.all(above_onset[i - min_duration_frames:i]):
                    end_idx = i
                    break
        
        return onset_idx, end_idx


# ============================================================================
# Otsu Adaptive Thresholding
# ============================================================================

def otsu_threshold(signal: np.ndarray, n_bins: int = 256) -> float:
    """
    Compute Otsu's threshold for a 1D signal.
    
    Otsu's method finds the threshold that minimizes intra-class variance,
    effectively separating foreground (voiding) from background (noise).
    
    Args:
        signal: 1D signal (e.g., RMS energy)
        n_bins: Number of histogram bins
        
    Returns:
        Optimal threshold value
    """
    # Normalize signal to [0, 1] for stable computation
    sig_min, sig_max = np.min(signal), np.max(signal)
    if sig_max - sig_min < 1e-10:
        return sig_min
    
    normalized = (signal - sig_min) / (sig_max - sig_min)
    
    # Compute histogram
    hist, bin_edges = np.histogram(normalized, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Total weight and mean
    total = len(normalized)
    sum_total = np.sum(bin_centers * hist)
    
    # Find optimal threshold
    best_threshold = 0
    best_variance = 0
    sum_background = 0
    weight_background = 0
    
    for i in range(n_bins):
        weight_background += hist[i]
        if weight_background == 0:
            continue
            
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += bin_centers[i] * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Between-class variance
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[i]
    
    # Convert back to original scale
    return best_threshold * (sig_max - sig_min) + sig_min


# ============================================================================
# Hybrid Detection (Otsu + Changepoint)
# ============================================================================

@dataclass
class AlternativeDetectionResult:
    """Result from alternative detection method."""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    voiding_time: float
    otsu_threshold: float
    method: str = "otsu_changepoint"


def detect_voiding_alternative(
    energy: np.ndarray,
    time_axis: np.ndarray,
    min_duration_ms: float = 300,
    hop_length_ms: float = 25,
    detector: Optional[ChangePointDetector] = None
) -> AlternativeDetectionResult:
    """
    Hybrid onset/end detection using Otsu thresholding and changepoint detection.
    
    This method provides an alternative to fixed-threshold detection,
    adapting to the actual energy distribution in the recording.
    
    Algorithm:
    1. Apply Otsu's method to derive adaptive threshold
    2. Use changepoint detector to find sustained transitions
    3. Onset: first changepoint where energy exceeds Otsu threshold
    4. End: last changepoint where energy falls below threshold
    5. Enforce minimum duration constraint
    
    When this may outperform fixed thresholds:
    - Recordings with varying background noise levels
    - Very quiet or very loud voiding sounds
    - Non-standard recording environments
    
    Args:
        energy: RMS energy signal
        time_axis: Time values for each frame
        min_duration_ms: Minimum voiding duration in milliseconds
        hop_length_ms: Frame hop length in milliseconds
        detector: Optional custom changepoint detector (for extensibility)
        
    Returns:
        AlternativeDetectionResult with detected times and parameters
    """
    # Calculate frame parameters
    min_duration_frames = int(min_duration_ms / hop_length_ms)
    
    # Step 1: Otsu adaptive threshold
    threshold = otsu_threshold(energy)
    
    # Step 2: Use provided detector or create default
    if detector is None:
        detector = RollingStatsChangePointDetector()
    
    # Step 3: Find onset and end using changepoint analysis
    onset_idx, end_idx = detector.detect_onset_end(energy, min_duration_frames)
    
    # Fallback: use Otsu threshold directly if changepoint fails
    if onset_idx is None or end_idx is None:
        above_threshold = energy > threshold
        
        if not np.any(above_threshold):
            # No voiding detected - return full range
            return AlternativeDetectionResult(
                start_idx=0,
                end_idx=len(energy) - 1,
                start_time=time_axis[0],
                end_time=time_axis[-1],
                voiding_time=float(time_axis[-1] - time_axis[0]),
                otsu_threshold=threshold
            )
        
        # Find first and last above threshold
        indices = np.where(above_threshold)[0]
        onset_idx = indices[0]
        end_idx = indices[-1]
    
    # Ensure valid indices
    onset_idx = max(0, onset_idx)
    end_idx = min(len(energy) - 1, end_idx)
    
    # Ensure onset is before end
    if onset_idx >= end_idx:
        onset_idx = 0
        end_idx = len(energy) - 1
    
    # Calculate times
    start_time = float(time_axis[onset_idx])
    end_time = float(time_axis[end_idx])
    voiding_time = end_time - start_time
    
    return AlternativeDetectionResult(
        start_idx=onset_idx,
        end_idx=end_idx,
        start_time=start_time,
        end_time=end_time,
        voiding_time=voiding_time,
        otsu_threshold=threshold
    )


# ============================================================================
# Future Extension Point: PELT Detector (using ruptures)
# ============================================================================

# To add ruptures-based PELT detection in the future:
#
# class RupturesPELTDetector:
#     def __init__(self, model: str = "rbf", min_size: int = 10, penalty: float = 1.0):
#         import ruptures as rpt
#         self.model = model
#         self.min_size = min_size
#         self.penalty = penalty
#
#     def detect(self, signal: np.ndarray, min_size: int = 10) -> list[int]:
#         import ruptures as rpt
#         algo = rpt.Pelt(model=self.model, min_size=min_size).fit(signal)
#         return algo.predict(pen=self.penalty)
#
# Usage:
#   detector = RupturesPELTDetector()
#   result = detect_voiding_alternative(energy, time_axis, detector=detector)

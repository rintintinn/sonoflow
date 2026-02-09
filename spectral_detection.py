"""
Spectral Voiding Detection for Acoustic Uroflowmetry

Parallel analysis module that detects voiding onset/offset using spectral
features rather than energy alone. Intended for research comparison alongside
the primary energy-based detection pipeline.

IMPORTANT — Input Audio:
    This module requires RAW (pre-bandpass) normalized audio, NOT the bandpass-
    filtered output from Step 2 of the processor pipeline. The reason:
    
    - The bandpass filter (250-4000 Hz) removes energy outside the voiding band
    - After filtering, ALL remaining energy is in the voiding band by definition
    - Band Energy Ratio becomes meaningless (~0.5-0.7 even during silence)
    - Spectral Centroid stays within voiding range even during silence
    
    Using raw audio preserves the full spectrum so the features can distinguish
    voiding (energy concentrated in 500-2500 Hz) from ambient noise (energy
    spread across all frequencies or concentrated at low/high extremes).

Energy Gating:
    Spectral features are unreliable when total frame energy is near the noise
    floor. During silence, the spectrum is just noise and produces meaningless
    centroid/BER/flatness values. Frames below an energy threshold have their
    voiding likelihood forced to zero, preventing false positives during silence.

Spectral Features Computed (per frame):
    1. Spectral Centroid — frequency centre of mass (Hz)
       - Noise: low centroid (room rumble) or very high (hiss)
       - Voiding: centroid in 800-2000 Hz range (splash broadband)

    2. Band Energy Ratio (BER) — energy in voiding band / total energy
       - Voiding band: 500-2500 Hz (core splash frequencies)
       - Noise: energy distributed outside this band
       - Voiding: BER > 0.5 (majority of energy in voiding band)

    3. Spectral Flatness — geometric mean / arithmetic mean of spectrum
       - Pure tone: flatness -> 0
       - White noise: flatness -> 1
       - Voiding: moderate flatness 0.1-0.5 (broadband but shaped)

    4. Voiding Likelihood Score — weighted combination of above features,
       gated by frame energy. Normalized 0-1.

Algorithm:
    1. Compute STFT on RAW audio with same frame/hop as energy extraction
    2. Extract spectral features per frame
    3. Compute per-frame RMS energy for gating
    4. Compute voiding likelihood score per frame
    5. Apply energy gate: zero out likelihood for low-energy frames
    6. Onset: first frame where gated score exceeds threshold for sustained period
    7. Offset: last frame where gated score exceeds threshold for sustained period

References:
    - Alvarez et al. 2025: spectral features in 0-8 kHz range for uroflow
    - Lee et al. 2021: ML spectral approach (r=0.88 for male Qmax)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpectralDetectionResult:
    """Result from spectral voiding detection."""
    # Detected onset/offset
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    voiding_time: float

    # Per-frame spectral features (aligned with energy time_axis)
    spectral_centroid: np.ndarray     # Hz, per frame
    band_energy_ratio: np.ndarray     # 0-1, per frame
    spectral_flatness: np.ndarray     # 0-1, per frame
    voiding_likelihood: np.ndarray    # 0-1, composite score per frame (energy-gated)
    frame_rms: np.ndarray             # Per-frame RMS energy (from raw audio STFT)

    # Detection parameters used
    likelihood_threshold: float
    energy_gate_threshold: float      # RMS threshold used for gating
    method: str = "spectral"


def compute_spectral_features(
    audio: np.ndarray,
    sample_rate: int,
    frame_length_ms: float = 50,
    frame_overlap: float = 0.5,
    voiding_band_low: float = 500.0,
    voiding_band_high: float = 2500.0,
    analysis_freq_limit: float = 6000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-frame spectral features from RAW (pre-bandpass) audio.

    IMPORTANT: Pass raw normalized audio, NOT bandpass-filtered audio.
    The STFT provides its own frequency decomposition. Bandpass filtering
    before STFT removes the spectral contrast needed for discrimination.

    All spectral features (centroid, BER, flatness) are computed only over
    frequencies from DC-excluded up to analysis_freq_limit (default 6 kHz).
    This prevents high-frequency content (mic self-noise, electronics hiss,
    HVAC harmonics) from dominating the features:
    - Without limit (0-24 kHz at 48k SR): centroid ~8000 Hz even during voiding
    - With 6 kHz limit: centroid properly falls in 800-2000 Hz during voiding

    The 6 kHz limit is chosen because:
    - Voiding acoustic energy is concentrated in 250-4000 Hz
    - Alvarez et al. 2025 analysed 0-8 kHz and found relevant features below 6 kHz
    - Room noise below 6 kHz provides useful spectral contrast for discrimination
    - Above 6 kHz is mostly mic noise that doesn't help voiding detection

    Uses the same frame/hop parameters as the energy extraction pipeline
    so that spectral features are aligned with the RMS energy time axis.

    Args:
        audio: RAW normalized audio signal (pre-bandpass, post-amplitude-normalization)
        sample_rate: Sample rate in Hz
        frame_length_ms: Frame length in ms (default 50, same as processor)
        frame_overlap: Overlap fraction (default 0.5, same as processor)
        voiding_band_low: Lower bound of voiding frequency band (Hz)
        voiding_band_high: Upper bound of voiding frequency band (Hz)
        analysis_freq_limit: Upper frequency limit for all spectral analysis (Hz).
            Default 6000. Only FFT bins below this frequency are used for
            centroid, BER, and flatness computation. This prevents HF noise
            from dominating features at high sample rates (44.1/48 kHz).

    Returns:
        Tuple of (time_axis, spectral_centroid, band_energy_ratio,
                  spectral_flatness, frame_rms, frequencies)
        All feature arrays have the same length (number of frames).
    """
    frame_length = int(frame_length_ms * sample_rate / 1000)
    hop_length = int(frame_length * (1 - frame_overlap))

    # Hann window (standard for spectral analysis)
    window = np.hanning(frame_length)

    # Number of frames
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        empty = np.array([0.0])
        return empty, empty, empty, empty, empty, empty

    # Frequency axis
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)

    # Frequency band masks — all restricted to analysis_freq_limit
    # analysis_mask: the "full spectrum" for centroid/BER/flatness (DC excluded, <= limit)
    analysis_mask = (frequencies > 0) & (frequencies <= analysis_freq_limit)
    # voiding_mask: the voiding band within the analysis range
    voiding_mask = (frequencies >= voiding_band_low) & (frequencies <= voiding_band_high)

    # Pre-allocate feature arrays
    centroids = np.zeros(n_frames)
    ber = np.zeros(n_frames)
    flatness = np.zeros(n_frames)
    frame_rms = np.zeros(n_frames)
    time_axis = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        raw_frame = audio[start:end]
        frame = raw_frame * window

        # Per-frame RMS (from raw frame, for energy gating)
        frame_rms[i] = float(np.sqrt(np.mean(raw_frame ** 2)))

        # FFT magnitude spectrum (positive frequencies only)
        spectrum = np.abs(np.fft.rfft(frame))
        power_spectrum = spectrum ** 2

        # Time for this frame (centre of frame)
        time_axis[i] = (start + frame_length / 2) / sample_rate

        # --- Spectral Centroid ---
        # Computed over analysis range only (0 to analysis_freq_limit)
        total_magnitude = np.sum(spectrum[analysis_mask])
        if total_magnitude > 1e-10:
            centroids[i] = np.sum(frequencies[analysis_mask] * spectrum[analysis_mask]) / total_magnitude
        else:
            centroids[i] = 0.0

        # --- Band Energy Ratio ---
        # Voiding band energy as fraction of analysis range energy
        total_energy = np.sum(power_spectrum[analysis_mask])
        voiding_energy = np.sum(power_spectrum[voiding_mask])
        if total_energy > 1e-10:
            ber[i] = voiding_energy / total_energy
        else:
            ber[i] = 0.0

        # --- Spectral Flatness ---
        # Computed over analysis range only
        ps_positive = power_spectrum[analysis_mask]
        ps_positive = ps_positive[ps_positive > 0]
        if len(ps_positive) > 0:
            log_mean = np.mean(np.log(ps_positive + 1e-20))
            arith_mean = np.mean(ps_positive)
            if arith_mean > 1e-20:
                flatness[i] = np.exp(log_mean) / arith_mean
            else:
                flatness[i] = 0.0
        else:
            flatness[i] = 0.0

    return time_axis, centroids, ber, flatness, frame_rms, frequencies


def compute_voiding_likelihood(
    spectral_centroid: np.ndarray,
    band_energy_ratio: np.ndarray,
    spectral_flatness: np.ndarray,
    # Expected ranges for voiding signature
    centroid_low: float = 600.0,
    centroid_high: float = 2200.0,
    centroid_peak: float = 1200.0,
    ber_weight: float = 0.45,
    centroid_weight: float = 0.35,
    flatness_weight: float = 0.20,
    # Flatness range for voiding (broadband but shaped)
    flatness_low: float = 0.02,
    flatness_high: float = 0.50,
    flatness_peak: float = 0.15,
) -> np.ndarray:
    """
    Compute per-frame voiding likelihood score from spectral features.

    Combines three spectral features into a single 0-1 score indicating
    how likely each frame contains voiding sound. This is the RAW likelihood
    before energy gating is applied.

    The scoring uses bell-curve / sigmoid shapes for each feature:
    - Centroid: peaks at ~1200 Hz, falls off below 600 and above 2200
    - BER: sigmoid — higher BER = more likely voiding
    - Flatness: peaks at ~0.15, falls off for pure tones (0) or white noise (1)

    Args:
        spectral_centroid: Per-frame centroid in Hz
        band_energy_ratio: Per-frame BER (0-1)
        spectral_flatness: Per-frame flatness (0-1)
        centroid_low/high/peak: Centroid scoring parameters
        ber_weight/centroid_weight/flatness_weight: Feature weights (sum to 1)
        flatness_low/high/peak: Flatness scoring parameters

    Returns:
        voiding_likelihood: Per-frame score 0-1 (before energy gating)
    """
    n_frames = len(spectral_centroid)
    scores = np.zeros(n_frames)

    for i in range(n_frames):
        # --- Centroid score (bell curve centred on voiding range) ---
        c = spectral_centroid[i]
        if c < centroid_low:
            c_score = max(0, c / centroid_low) if centroid_low > 0 else 0
        elif c > centroid_high:
            decay_range = centroid_high * 0.5
            c_score = max(0, 1.0 - (c - centroid_high) / decay_range)
        else:
            sigma = (centroid_high - centroid_low) / 4
            c_score = np.exp(-0.5 * ((c - centroid_peak) / sigma) ** 2)

        # --- BER score (sigmoid — higher is better) ---
        b = band_energy_ratio[i]
        b_score = 1.0 / (1.0 + np.exp(-15 * (b - 0.35)))

        # --- Flatness score (bell curve — voiding is moderately flat) ---
        f = spectral_flatness[i]
        if f < flatness_low:
            f_score = max(0, f / flatness_low) if flatness_low > 0 else 0
        elif f > flatness_high:
            f_score = max(0, 1.0 - (f - flatness_high) / (1.0 - flatness_high))
        else:
            sigma_f = (flatness_high - flatness_low) / 3
            f_score = np.exp(-0.5 * ((f - flatness_peak) / sigma_f) ** 2)

        # Weighted combination
        scores[i] = (
            centroid_weight * c_score
            + ber_weight * b_score
            + flatness_weight * f_score
        )

    return np.clip(scores, 0, 1)


def _compute_energy_gate(
    frame_rms: np.ndarray,
    energy_gate_percentile: float = 15,
    energy_gate_mult: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """
    Compute energy gate mask and threshold.

    Frames with RMS energy below the gate threshold are considered silence
    and should have their voiding likelihood zeroed out.

    The gate threshold is set at energy_gate_mult × noise floor, where
    noise floor is estimated as the energy_gate_percentile of frame RMS.
    This is similar to the main pipeline's noise floor estimation.

    Args:
        frame_rms: Per-frame RMS energy
        energy_gate_percentile: Percentile for noise floor (default 15)
        energy_gate_mult: Multiplier above noise floor (default 3.0)

    Returns:
        Tuple of (gate_mask, gate_threshold)
        gate_mask: boolean array, True = frame has sufficient energy
        gate_threshold: the RMS threshold used
    """
    noise_floor = float(np.percentile(frame_rms, energy_gate_percentile))
    gate_threshold = noise_floor * energy_gate_mult

    # Soft gating: use a sigmoid transition instead of hard cutoff
    # This prevents abrupt on/off artefacts at the gate boundary
    # sigmoid centered at gate_threshold, width ~20% of threshold
    sigmoid_width = max(gate_threshold * 0.3, 1e-10)
    gate_weights = 1.0 / (1.0 + np.exp(-(frame_rms - gate_threshold) / sigmoid_width))

    return gate_weights, gate_threshold


def detect_voiding_spectral(
    audio: np.ndarray,
    sample_rate: int,
    energy_time_axis: np.ndarray,
    frame_length_ms: float = 50,
    frame_overlap: float = 0.5,
    # Detection parameters
    likelihood_threshold: float = 0.25,
    min_sustained_sec: float = 0.3,
    # Energy gating parameters
    energy_gate_percentile: float = 15,
    energy_gate_mult: float = 1.5,
    # Smoothing
    smooth_window: int = 5,
) -> SpectralDetectionResult:
    """
    Detect voiding onset/offset using spectral features with energy gating.

    Runs as a parallel analysis module — does not modify the primary
    energy-based detection. Results are for research comparison only.

    IMPORTANT: Pass RAW normalized audio (post Step 1, pre Step 2).
    Do NOT pass bandpass-filtered audio — the filtering removes the spectral
    contrast needed for voiding vs noise discrimination.

    Args:
        audio: RAW normalized audio (pre-bandpass). This is the output of
            processor._normalize_input(), before _bandpass_filter().
        sample_rate: Sample rate in Hz
        energy_time_axis: Time axis from the energy extraction step
            (used to align spectral frames with energy frames)
        frame_length_ms: Frame length (same as processor for alignment)
        frame_overlap: Overlap fraction (same as processor for alignment)
        likelihood_threshold: Minimum voiding likelihood for detection.
            Default 0.4 (moderate confidence).
        min_sustained_sec: Minimum sustained duration above threshold.
            Default 0.3s.
        energy_gate_percentile: Percentile for noise floor estimation in
            energy gating. Default 15.
        energy_gate_mult: Multiplier above noise floor for energy gate.
            Default 3.0. Frames below this threshold have likelihood zeroed.
        smooth_window: Window size for likelihood smoothing (frames).
            Default 5 (~125ms).

    Returns:
        SpectralDetectionResult with detected times and per-frame features.
    """
    # Step 1: Compute spectral features from RAW audio
    spec_time, centroids, ber, flatness, frame_rms, freqs = compute_spectral_features(
        audio, sample_rate, frame_length_ms, frame_overlap
    )

    # Step 2: Compute raw voiding likelihood (before energy gating)
    likelihood_raw = compute_voiding_likelihood(centroids, ber, flatness)

    # Step 3: Compute energy gate
    gate_weights, gate_threshold = _compute_energy_gate(
        frame_rms, energy_gate_percentile, energy_gate_mult
    )

    # Step 4: Apply energy gate — multiply likelihood by gate weights
    # During silence: gate_weights ≈ 0, so likelihood → 0
    # During voiding: gate_weights ≈ 1, so likelihood unchanged
    likelihood_gated = likelihood_raw * gate_weights

    # Step 5: Smooth gated likelihood
    if len(likelihood_gated) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        likelihood_smooth = np.convolve(likelihood_gated, kernel, mode="same")
    else:
        likelihood_smooth = likelihood_gated.copy()

    # --- Align spectral frames with energy frames ---
    if len(spec_time) != len(energy_time_axis):
        centroids = np.interp(energy_time_axis, spec_time, centroids)
        ber = np.interp(energy_time_axis, spec_time, ber)
        flatness = np.interp(energy_time_axis, spec_time, flatness)
        frame_rms = np.interp(energy_time_axis, spec_time, frame_rms)
        likelihood_smooth = np.interp(energy_time_axis, spec_time, likelihood_smooth)
        spec_time = energy_time_axis

    # --- Detect onset: first sustained period above threshold ---
    hop_sec = float(spec_time[1] - spec_time[0]) if len(spec_time) > 1 else 0.025
    min_sustained_frames = int(min_sustained_sec / hop_sec)

    above = likelihood_smooth >= likelihood_threshold

    start_idx = 0
    found_start = False
    for i in range(len(above) - min_sustained_frames):
        if np.all(above[i : i + min_sustained_frames]):
            start_idx = i
            found_start = True
            break

    # --- Detect offset: last sustained period above threshold ---
    end_idx = len(above) - 1
    found_end = False
    for i in range(len(above) - 1, min_sustained_frames, -1):
        if np.all(above[i - min_sustained_frames : i]):
            end_idx = i
            found_end = True
            break

    # If neither found, no voiding detected
    if not found_start and not found_end:
        # Return zero-duration result at midpoint
        mid = len(spec_time) // 2
        return SpectralDetectionResult(
            start_idx=mid,
            end_idx=mid,
            start_time=float(spec_time[mid]),
            end_time=float(spec_time[mid]),
            voiding_time=0.0,
            spectral_centroid=centroids,
            band_energy_ratio=ber,
            spectral_flatness=flatness,
            voiding_likelihood=likelihood_smooth,
            frame_rms=frame_rms,
            likelihood_threshold=likelihood_threshold,
            energy_gate_threshold=gate_threshold,
        )

    # Ensure valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(spec_time) - 1, end_idx)
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = len(spec_time) - 1

    start_time = float(spec_time[start_idx])
    end_time = float(spec_time[end_idx])
    voiding_time = end_time - start_time

    return SpectralDetectionResult(
        start_idx=start_idx,
        end_idx=end_idx,
        start_time=start_time,
        end_time=end_time,
        voiding_time=voiding_time,
        spectral_centroid=centroids,
        band_energy_ratio=ber,
        spectral_flatness=flatness,
        voiding_likelihood=likelihood_smooth,
        frame_rms=frame_rms,
        likelihood_threshold=likelihood_threshold,
        energy_gate_threshold=gate_threshold,
    )

"""
Multi-Episode Voiding Detection for Acoustic Uroflowmetry

Handles intermittent voiding patterns (straining, BPH) where the urine stream
stops and restarts multiple times during a single voiding event.

Uses Otsu adaptive thresholding + rolling-statistics changepoint detection
as the primary detection method, consistent with the project's default
detection pipeline (alternative_detection.py).

Algorithm:
    1. Compute adaptive threshold via Otsu's method (bimodal separation)
    2. Derive a scanning threshold (lower than Otsu) for episode boundary detection
    3. Detect all candidate episodes (contiguous above-scanning-threshold regions)
    4. Filter out non-voiding noise spikes (too short or too weak)
    5. Classify gaps between episodes (short=straining, long=post-void)
    6. Merge included episodes into a single voiding window
    7. Refine episode edges using changepoint detector

Why two thresholds:
    - Otsu gives the optimal separation between noise and voiding energy classes,
      but variable-energy episodes (common in BPH) have many dips below Otsu,
      fragmenting them into dozens of micro-episodes that get rejected.
    - The scanning threshold (midpoint between noise floor and Otsu) captures
      the broad envelope of each episode, while Otsu is used to validate that
      detected episodes contain genuine voiding energy.

References:
    - Straining patterns are common in BPH patients and can produce
      3-8 separate flow episodes within a single voiding event
    - ICS defines voiding time as including intermittent pauses
      (flow time excludes them, voiding time includes them)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from alternative_detection import (
    otsu_threshold,
    RollingStatsChangePointDetector,
    ChangePointDetector,
)


@dataclass
class FlowEpisode:
    """A single contiguous flow episode within a voiding event."""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    peak_energy: float
    mean_energy: float


@dataclass
class MultiEpisodeResult:
    """Result from multi-episode voiding detection."""
    # Overall voiding window (from first episode start to last episode end)
    voiding_start_idx: int
    voiding_end_idx: int
    voiding_start_time: float
    voiding_end_time: float
    voiding_time: float          # Total time including pauses (ICS "voiding time")
    flow_time: float             # Time of actual flow only, excluding pauses (ICS "flow time")

    # Individual episodes
    episodes: List[FlowEpisode]
    num_episodes: int

    # Flow pattern classification
    pattern: str                 # "continuous", "intermittent", "straining"

    # Gap information
    gap_durations: List[float]   # Duration of each inter-episode gap

    # Detection metadata
    otsu_thresh: float           # Otsu threshold (for classification/validation)
    scan_thresh: float           # Scanning threshold used for episode detection
    method: str = "otsu_changepoint_multiepisode"

    # Indices of episodes that were rejected (for debugging)
    rejected_episodes: List[FlowEpisode] = field(default_factory=list)


def detect_voiding_multiepisode(
    energy: np.ndarray,
    time_axis: np.ndarray,
    noise_floor: Optional[float] = None,
    # Episode filtering parameters
    min_episode_duration_sec: float = 0.3,
    min_episode_peak_fraction: float = 0.05,
    # Gap classification parameters
    short_gap_max_sec: float = 2.0,
    long_gap_min_sec: float = 5.0,
    medium_gap_energy_fraction: float = 0.20,
    # Scanning threshold control
    scan_threshold_fraction: float = 0.5,
    # Changepoint detector (optional override)
    detector: Optional[ChangePointDetector] = None,
    # Otsu parameters
    otsu_n_bins: int = 256,
) -> MultiEpisodeResult:
    """
    Detect voiding segment with support for intermittent/straining patterns.

    Uses a two-threshold approach:
    - Otsu threshold: optimal noise/voiding separation (for episode validation)
    - Scanning threshold: lower threshold for finding episode boundaries
      (captures the broad envelope without fragmenting variable-energy episodes)

    Args:
        energy: RMS energy signal (from Step 3 of pipeline)
        time_axis: Time values for each energy frame
        noise_floor: Optional noise floor estimate. If None, uses 10th percentile
            of energy (same as processor.py Step 4).

        min_episode_duration_sec: Minimum episode duration to be considered real flow.
            Shorter episodes are rejected as noise spikes. Default 0.3s.
        min_episode_peak_fraction: Minimum peak energy of an episode relative to
            the global peak. Episodes weaker than this are rejected. Default 0.05 (5%).

        short_gap_max_sec: Gaps shorter than this are always included (straining pause).
            Default 2.0s.
        long_gap_min_sec: Gaps longer than this are always excluded (post-void).
            Default 5.0s.
        medium_gap_energy_fraction: For gaps between short and long, include the next
            episode only if its peak exceeds this fraction of global peak. Default 0.20 (20%).

        scan_threshold_fraction: Where to set the scanning threshold between noise_floor
            and Otsu threshold. 0.0 = noise_floor, 1.0 = Otsu, 0.5 = midpoint (default).
            Lower values capture more of the episode envelope but may include more noise.

        detector: Optional custom changepoint detector. If None, uses the default
            RollingStatsChangePointDetector (same as alternative_detection.py).
        otsu_n_bins: Number of histogram bins for Otsu's method. Default 256.

    Returns:
        MultiEpisodeResult with detected episodes, merged voiding window,
        and pattern classification.
    """

    # Compute hop time for frame-to-time conversions
    if len(time_axis) > 1:
        hop_sec = float(time_axis[1] - time_axis[0])
    else:
        hop_sec = 0.025  # Default 25ms

    # =========================================================================
    # Step 1: Compute adaptive threshold via Otsu's method
    # =========================================================================

    otsu_thresh = otsu_threshold(energy, n_bins=otsu_n_bins)

    # Estimate noise floor if not provided
    if noise_floor is None:
        noise_floor = float(np.percentile(energy, 10))

    # =========================================================================
    # Step 2: Derive scanning threshold
    # =========================================================================
    #
    # The scanning threshold is set between noise_floor and Otsu threshold.
    # This is lower than Otsu, so it captures the broad envelope of variable-
    # energy episodes without fragmenting them into micro-episodes.
    #
    # The key insight: Otsu is optimal for binary classification (noise vs voiding)
    # but too aggressive for episode boundary detection when voiding energy is
    # highly variable (BPH straining, intermittent flow). A lower threshold
    # finds the gross episode boundaries, while Otsu validates their content.
    # =========================================================================

    scan_thresh = noise_floor + scan_threshold_fraction * (otsu_thresh - noise_floor)

    # Safety: ensure scanning threshold is at least above noise floor
    scan_thresh = max(scan_thresh, noise_floor * 1.5)

    # Initialize changepoint detector for edge refinement (Step 7)
    if detector is None:
        detector = RollingStatsChangePointDetector()

    # =========================================================================
    # Step 3: Find ALL candidate episodes (contiguous above-scanning-threshold)
    # =========================================================================

    above_threshold = energy > scan_thresh

    if not np.any(above_threshold):
        # No flow detected — return entire recording as fallback
        return _fallback_result(energy, time_axis, otsu_thresh, scan_thresh)

    # Find contiguous regions above threshold
    padded = np.concatenate([[False], above_threshold, [False]])
    diffs = np.diff(padded.astype(int))

    episode_starts = np.where(diffs == 1)[0]   # Rising edges
    episode_ends = np.where(diffs == -1)[0]     # Falling edges

    # Build raw candidate episodes
    global_peak = np.max(energy)
    min_episode_frames = int(min_episode_duration_sec / hop_sec)

    raw_episodes = []
    for start, end in zip(episode_starts, episode_ends):
        end_inclusive = end - 1

        if end_inclusive < start:
            continue

        ep_energy = energy[start:end]
        raw_episodes.append(FlowEpisode(
            start_idx=int(start),
            end_idx=int(end_inclusive),
            start_time=float(time_axis[start]),
            end_time=float(time_axis[end_inclusive]),
            duration=float(time_axis[end_inclusive] - time_axis[start]),
            peak_energy=float(np.max(ep_energy)),
            mean_energy=float(np.mean(ep_energy)),
        ))

    if len(raw_episodes) == 0:
        return _fallback_result(energy, time_axis, otsu_thresh, scan_thresh)

    # =========================================================================
    # Step 4: Filter out non-voiding episodes
    # =========================================================================
    #
    # Two-stage filter:
    #   a) Reject episodes too short (noise spikes)
    #   b) Reject episodes whose PEAK is below min_episode_peak_fraction of global peak
    #   c) Validate: episode must have at least some frames above Otsu threshold
    #      (confirms it contains genuine voiding energy, not just elevated noise)
    # =========================================================================

    valid_episodes = []
    rejected_episodes = []

    for ep in raw_episodes:
        ep_frames = ep.end_idx - ep.start_idx + 1

        # Reject if too short (noise spike, cough, door slam)
        if ep_frames < min_episode_frames:
            rejected_episodes.append(ep)
            continue

        # Reject if too weak relative to global peak
        if ep.peak_energy < global_peak * min_episode_peak_fraction:
            rejected_episodes.append(ep)
            continue

        # Validate: at least some content above Otsu threshold
        # (confirms this is a real voiding episode, not just elevated background)
        ep_above_otsu = np.sum(energy[ep.start_idx:ep.end_idx + 1] > otsu_thresh)
        otsu_fraction = ep_above_otsu / ep_frames
        if otsu_fraction < 0.1:  # Less than 10% of episode above Otsu → likely noise
            rejected_episodes.append(ep)
            continue

        valid_episodes.append(ep)

    if len(valid_episodes) == 0:
        # All episodes rejected — use the strongest raw episode as fallback
        strongest = max(raw_episodes, key=lambda e: e.peak_energy)
        valid_episodes = [strongest]

    # =========================================================================
    # Step 5: Classify gaps and decide which episodes to merge
    # =========================================================================

    included_episodes = [valid_episodes[0]]
    gap_durations = []

    for i in range(1, len(valid_episodes)):
        prev_ep = included_episodes[-1]
        curr_ep = valid_episodes[i]

        gap_duration = curr_ep.start_time - prev_ep.end_time
        gap_durations.append(gap_duration)

        include = False

        if gap_duration <= short_gap_max_sec:
            # Short gap: almost certainly a straining pause → include
            include = True

        elif gap_duration >= long_gap_min_sec:
            # Long gap: likely post-void activity → exclude
            include = False

        else:
            # Medium gap (2-5s): include if next episode is significant
            if curr_ep.peak_energy >= global_peak * medium_gap_energy_fraction:
                include = True
            else:
                include = False

        if include:
            included_episodes.append(curr_ep)
        else:
            # Stop including further episodes once we hit a long/rejected gap
            break

    # =========================================================================
    # Step 6: Merge adjacent/overlapping episodes
    # =========================================================================
    #
    # Because we use a lower scanning threshold, episodes that were separate
    # at the Otsu level may now be contiguous or overlapping. Merge them.
    # =========================================================================

    merged_episodes = [included_episodes[0]]
    for ep in included_episodes[1:]:
        prev = merged_episodes[-1]
        # If this episode starts within 1 frame of previous end, merge
        if ep.start_idx <= prev.end_idx + 2:
            # Merge: extend previous episode
            merged_energy = energy[prev.start_idx:ep.end_idx + 1]
            merged_episodes[-1] = FlowEpisode(
                start_idx=prev.start_idx,
                end_idx=ep.end_idx,
                start_time=prev.start_time,
                end_time=ep.end_time,
                duration=float(time_axis[ep.end_idx] - time_axis[prev.start_idx]),
                peak_energy=float(np.max(merged_energy)),
                mean_energy=float(np.mean(merged_energy)),
            )
        else:
            merged_episodes.append(ep)

    included_episodes = merged_episodes

    # =========================================================================
    # Step 7: Refine edges using changepoint detector
    # =========================================================================

    first_ep = included_episodes[0]
    last_ep = included_episodes[-1]

    changepoints = detector.detect(energy, min_size=max(5, int(0.1 / hop_sec)))

    # Refine start: find the nearest changepoint before first episode
    refined_start = first_ep.start_idx
    if changepoints:
        onset_candidates = [cp for cp in changepoints if cp <= first_ep.start_idx]
        if onset_candidates:
            best_cp = max(onset_candidates)
            if (first_ep.start_idx - best_cp) * hop_sec <= 1.0:
                refined_start = best_cp

    # Refine end: find the nearest changepoint after last episode
    refined_end = last_ep.end_idx
    if changepoints:
        end_candidates = [cp for cp in changepoints if cp >= last_ep.end_idx]
        if end_candidates:
            best_cp = min(end_candidates)
            if (best_cp - last_ep.end_idx) * hop_sec <= 1.0:
                refined_end = best_cp

    # Clamp to valid range
    refined_start = max(0, refined_start)
    refined_end = min(len(energy) - 1, refined_end)

    if refined_start >= refined_end:
        refined_start = first_ep.start_idx
        refined_end = last_ep.end_idx

    # =========================================================================
    # Compute timing metrics
    # =========================================================================

    voiding_start_time = float(time_axis[refined_start])
    voiding_end_time = float(time_axis[refined_end])
    voiding_time = voiding_end_time - voiding_start_time

    # Flow time = sum of episode durations only (excluding inter-episode gaps)
    flow_time = sum(ep.duration for ep in included_episodes)

    # =========================================================================
    # Classify flow pattern
    # =========================================================================

    if len(included_episodes) == 1:
        pattern = "continuous"
    elif len(included_episodes) <= 3:
        pattern = "intermittent"
    else:
        pattern = "straining"

    return MultiEpisodeResult(
        voiding_start_idx=int(refined_start),
        voiding_end_idx=int(refined_end),
        voiding_start_time=voiding_start_time,
        voiding_end_time=voiding_end_time,
        voiding_time=voiding_time,
        flow_time=flow_time,
        episodes=included_episodes,
        num_episodes=len(included_episodes),
        pattern=pattern,
        gap_durations=gap_durations,
        otsu_thresh=float(otsu_thresh),
        scan_thresh=float(scan_thresh),
        rejected_episodes=rejected_episodes,
    )


def _fallback_result(
    energy: np.ndarray,
    time_axis: np.ndarray,
    otsu_thresh: float,
    scan_thresh: float,
) -> MultiEpisodeResult:
    """Fallback when no flow is detected — return entire recording."""
    total_time = float(time_axis[-1] - time_axis[0]) if len(time_axis) > 1 else 0.0

    fallback_episode = FlowEpisode(
        start_idx=0,
        end_idx=len(energy) - 1,
        start_time=float(time_axis[0]),
        end_time=float(time_axis[-1]),
        duration=total_time,
        peak_energy=float(np.max(energy)) if len(energy) > 0 else 0.0,
        mean_energy=float(np.mean(energy)) if len(energy) > 0 else 0.0,
    )

    return MultiEpisodeResult(
        voiding_start_idx=0,
        voiding_end_idx=len(energy) - 1,
        voiding_start_time=float(time_axis[0]),
        voiding_end_time=float(time_axis[-1]),
        voiding_time=total_time,
        flow_time=total_time,
        episodes=[fallback_episode],
        num_episodes=1,
        pattern="continuous",
        gap_durations=[],
        otsu_thresh=otsu_thresh,
        scan_thresh=scan_thresh,
    )

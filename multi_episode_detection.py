"""
Multi-Episode Voiding Detection for Acoustic Uroflowmetry

Handles intermittent voiding patterns (straining, BPH) where the urine stream
stops and restarts multiple times during a single voiding event.

The standard single-pass end-detection terminates at the first sustained silence,
missing subsequent flow episodes. This module detects ALL episodes first, then
merges them based on gap duration and energy criteria.

Algorithm:
    1. Detect all candidate episodes (contiguous above-threshold regions)
    2. Filter out non-voiding noise spikes (too short or too weak)
    3. Classify gaps between episodes (short=straining, long=post-void)
    4. Merge included episodes into a single voiding window

References:
    - Straining patterns are common in BPH patients and can produce
      3-8 separate flow episodes within a single voiding event
    - ICS defines voiding time as including intermittent pauses
      (flow time excludes them, voiding time includes them)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


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
    
    # Indices of episodes that were rejected (for debugging)
    rejected_episodes: List[FlowEpisode] = field(default_factory=list)


def detect_voiding_multiepisode(
    energy: np.ndarray,
    time_axis: np.ndarray,
    noise_floor: float,
    # Episode detection parameters
    onset_threshold_mult: float = 3.0,
    min_episode_duration_sec: float = 0.3,
    min_episode_peak_fraction: float = 0.05,
    # Gap classification parameters
    short_gap_max_sec: float = 2.0,
    long_gap_min_sec: float = 5.0,
    medium_gap_energy_fraction: float = 0.20,
    # Refinement parameters
    edge_threshold_mult: float = 1.5,
    edge_walkback_frames: int = 40,
) -> MultiEpisodeResult:
    """
    Detect voiding segment with support for intermittent/straining patterns.
    
    Args:
        energy: RMS energy signal (from Step 3 of pipeline)
        time_axis: Time values for each energy frame
        noise_floor: Estimated noise floor (from Step 4)
        
        onset_threshold_mult: Multiplier above noise floor for episode detection.
            Default 3.0 (consistent with original fixed-threshold method).
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
            
        edge_threshold_mult: Multiplier for edge refinement (walk back/forward from
            episode boundaries). Default 1.5.
        edge_walkback_frames: Maximum frames to walk back for edge refinement. Default 40.
    
    Returns:
        MultiEpisodeResult with detected episodes, merged voiding window, and pattern classification.
    """
    
    # Compute hop time for frame-to-time conversions
    if len(time_axis) > 1:
        hop_sec = float(time_axis[1] - time_axis[0])
    else:
        hop_sec = 0.025  # Default 25ms
    
    # =========================================================================
    # Step 1: Find ALL candidate episodes (contiguous above-threshold regions)
    # =========================================================================
    
    flow_threshold = noise_floor * onset_threshold_mult
    above_threshold = energy > flow_threshold
    
    if not np.any(above_threshold):
        # No flow detected — return entire recording as fallback
        return _fallback_result(energy, time_axis)
    
    # Find contiguous regions above threshold
    # Detect transitions: 0→1 = episode start, 1→0 = episode end
    padded = np.concatenate([[False], above_threshold, [False]])
    diffs = np.diff(padded.astype(int))
    
    episode_starts = np.where(diffs == 1)[0]   # Rising edges
    episode_ends = np.where(diffs == -1)[0]     # Falling edges
    
    # Build raw candidate episodes
    global_peak = np.max(energy)
    min_episode_frames = int(min_episode_duration_sec / hop_sec)
    
    raw_episodes = []
    for start, end in zip(episode_starts, episode_ends):
        # end is exclusive (first frame below threshold), so episode is [start, end-1]
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
        return _fallback_result(energy, time_axis)
    
    # =========================================================================
    # Step 2: Filter out non-voiding episodes
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
        
        valid_episodes.append(ep)
    
    if len(valid_episodes) == 0:
        # All episodes rejected — use the strongest raw episode as fallback
        strongest = max(raw_episodes, key=lambda e: e.peak_energy)
        valid_episodes = [strongest]
    
    # =========================================================================
    # Step 3: Classify gaps and decide which episodes to merge
    # =========================================================================
    
    # Start with the first valid episode, then decide whether to include subsequent ones
    included_episodes = [valid_episodes[0]]
    gap_durations = []
    
    for i in range(1, len(valid_episodes)):
        prev_ep = included_episodes[-1]
        curr_ep = valid_episodes[i]
        
        gap_duration = curr_ep.start_time - prev_ep.end_time
        gap_durations.append(gap_duration)
        
        # Classification logic
        include = False
        
        if gap_duration <= short_gap_max_sec:
            # Short gap: almost certainly a straining pause → include
            include = True
            
        elif gap_duration >= long_gap_min_sec:
            # Long gap: likely post-void activity → exclude
            # (also excludes all subsequent episodes)
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
            # (episodes after a long gap are likely post-void activity)
            break
    
    # =========================================================================
    # Step 4: Refine edges and compute final voiding window
    # =========================================================================
    
    first_ep = included_episodes[0]
    last_ep = included_episodes[-1]
    
    # Refine start: walk back from first episode to capture upslope
    refined_start = first_ep.start_idx
    for i in range(first_ep.start_idx, max(first_ep.start_idx - edge_walkback_frames, 0), -1):
        if energy[i] < noise_floor * edge_threshold_mult:
            refined_start = i + 1
            break
    else:
        refined_start = max(first_ep.start_idx - edge_walkback_frames, 0)
    
    # Refine end: walk forward from last episode to capture tail
    refined_end = last_ep.end_idx
    for i in range(last_ep.end_idx, min(last_ep.end_idx + edge_walkback_frames, len(energy) - 1)):
        if energy[i] < noise_floor * edge_threshold_mult:
            refined_end = i
            break
    else:
        refined_end = min(last_ep.end_idx + edge_walkback_frames, len(energy) - 1)
    
    # Clamp to valid range
    refined_start = max(0, refined_start)
    refined_end = min(len(energy) - 1, refined_end)
    
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
        rejected_episodes=rejected_episodes,
    )


def _fallback_result(energy: np.ndarray, time_axis: np.ndarray) -> MultiEpisodeResult:
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
    )

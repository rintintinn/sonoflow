"""
Debug Visualization Module for Acoustic Uroflowmetry

Provides research and debugging plots for signal analysis:
- Raw waveform visualization with voiding markers
- Energy curve with thresholds and detection points
- Detection method comparison (fixed vs alternative)

These plots are intended for research and debugging only,
not as primary user-facing output.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from typing import Tuple, Optional


def plot_raw_waveform(
    filtered_audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    title: str = "Filtered Waveform with Voiding Markers"
) -> bytes:
    """
    Plot the band-pass filtered waveform with voiding start/end markers.
    
    Args:
        filtered_audio: Post-bandpass-filtered audio samples
        sample_rate: Sample rate in Hz
        start_time: Detected voiding start time (seconds)
        end_time: Detected voiding end time (seconds)
        title: Plot title
        
    Returns:
        PNG image as bytes
    """
    # Create time axis
    duration = len(filtered_audio) / sample_rate
    time_axis = np.linspace(0, duration, len(filtered_audio))
    
    # Normalize amplitude for display
    max_amp = np.max(np.abs(filtered_audio))
    normalized = filtered_audio / max_amp if max_amp > 0 else filtered_audio
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1e293b')
    ax.set_facecolor('#0f172a')
    
    # Plot waveform
    ax.plot(time_axis, normalized, color='#38bdf8', linewidth=0.5, alpha=0.8)
    
    # Add voiding markers
    ax.axvline(x=start_time, color='#22c55e', linestyle='--', linewidth=2, label='Voiding Start')
    ax.axvline(x=end_time, color='#ef4444', linestyle='--', linewidth=2, label='Voiding End')
    
    # Shade voiding region
    ax.axvspan(start_time, end_time, alpha=0.15, color='#22c55e')
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=10)
    ax.set_ylabel('Normalized Amplitude', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
    ax.grid(True, alpha=0.2, color='#475569')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='#1e293b', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def plot_energy_curve(
    time_axis: np.ndarray,
    energy: np.ndarray,
    noise_floor: float,
    start_idx: int,
    end_idx: int,
    threshold_mult: float = 3.0,
    title: str = "Energy Curve with Detection Thresholds"
) -> bytes:
    """
    Plot the RMS energy curve with noise floor and detection thresholds.
    
    Args:
        time_axis: Time values for each energy frame
        energy: RMS energy values
        noise_floor: Estimated noise floor
        start_idx: Detected voiding start index
        end_idx: Detected voiding end index
        threshold_mult: Multiple of noise floor used for detection
        title: Plot title
        
    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#1e293b')
    ax.set_facecolor('#0f172a')
    
    # Plot energy curve
    ax.plot(time_axis, energy, color='#38bdf8', linewidth=1.5, label='RMS Energy')
    
    # Horizontal threshold lines
    ax.axhline(y=noise_floor, color='#f59e0b', linestyle='-', linewidth=1.5, 
               label=f'Noise Floor ({noise_floor:.4f})')
    ax.axhline(y=noise_floor * threshold_mult, color='#ef4444', linestyle='--', linewidth=1.5,
               label=f'Detection Threshold ({threshold_mult}× noise)')
    
    # Vertical markers for start/end
    start_time = time_axis[start_idx]
    end_time = time_axis[end_idx]
    ax.axvline(x=start_time, color='#22c55e', linestyle=':', linewidth=2, label=f'Start ({start_time:.1f}s)')
    ax.axvline(x=end_time, color='#f43f5e', linestyle=':', linewidth=2, label=f'End ({end_time:.1f}s)')
    
    # Shade voiding region
    ax.axvspan(start_time, end_time, alpha=0.1, color='#22c55e')
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=10)
    ax.set_ylabel('RMS Energy', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', 
              labelcolor='white', fontsize=8)
    ax.grid(True, alpha=0.2, color='#475569')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='#1e293b', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def plot_detection_comparison(
    time_axis: np.ndarray,
    energy: np.ndarray,
    noise_floor: float,
    fixed_start_idx: int,
    fixed_end_idx: int,
    alt_start_idx: int,
    alt_end_idx: int,
    title: str = "Detection Method Comparison"
) -> bytes:
    """
    Plot comparison of fixed-threshold vs alternative detection methods.
    
    Args:
        time_axis: Time values for each energy frame
        energy: RMS energy values
        noise_floor: Estimated noise floor
        fixed_start_idx: Start index from fixed-threshold method
        fixed_end_idx: End index from fixed-threshold method
        alt_start_idx: Start index from alternative method
        alt_end_idx: End index from alternative method
        title: Plot title
        
    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#1e293b')
    ax.set_facecolor('#0f172a')
    
    # Plot energy curve
    ax.plot(time_axis, energy, color='#38bdf8', linewidth=1.5, label='RMS Energy', alpha=0.8)
    
    # Noise floor reference
    ax.axhline(y=noise_floor, color='#f59e0b', linestyle='-', linewidth=1, 
               label='Noise Floor', alpha=0.7)
    
    # Fixed-threshold method markers (solid lines)
    fixed_start_time = time_axis[fixed_start_idx]
    fixed_end_time = time_axis[fixed_end_idx]
    ax.axvline(x=fixed_start_time, color='#22c55e', linestyle='-', linewidth=2)
    ax.axvline(x=fixed_end_time, color='#22c55e', linestyle='-', linewidth=2)
    
    # Alternative method markers (dashed lines)
    alt_start_time = time_axis[alt_start_idx]
    alt_end_time = time_axis[alt_end_idx]
    ax.axvline(x=alt_start_time, color='#a855f7', linestyle='--', linewidth=2)
    ax.axvline(x=alt_end_time, color='#a855f7', linestyle='--', linewidth=2)
    
    # Create custom legend
    fixed_patch = mpatches.Patch(color='#22c55e', label=f'Otsu+Changepoint (default): {fixed_start_time:.1f}s – {fixed_end_time:.1f}s')
    alt_patch = mpatches.Patch(color='#a855f7', label=f'Legacy Fixed-Threshold: {alt_start_time:.1f}s – {alt_end_time:.1f}s')
    ax.legend(handles=[fixed_patch, alt_patch], loc='upper right', 
              facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=9)
    
    # Calculate and display time differences
    start_diff = alt_start_time - fixed_start_time
    end_diff = alt_end_time - fixed_end_time
    duration_fixed = fixed_end_time - fixed_start_time
    duration_alt = alt_end_time - alt_start_time
    
    info_text = f'Otsu (default): {duration_fixed:.1f}s | Legacy: {duration_alt:.1f}s | Δ={duration_alt - duration_fixed:+.1f}s'
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, color='white', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#334155', edgecolor='#475569', alpha=0.9))
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=10)
    ax.set_ylabel('RMS Energy', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='#475569')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='#1e293b', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def generate_all_debug_plots(
    filtered_audio: np.ndarray,
    sample_rate: int,
    time_axis: np.ndarray,
    energy: np.ndarray,
    noise_floor: float,
    fixed_start_idx: int,
    fixed_end_idx: int,
    alt_start_idx: Optional[int] = None,
    alt_end_idx: Optional[int] = None
) -> Tuple[bytes, bytes, Optional[bytes]]:
    """
    Generate all debug plots in one call.
    
    Returns:
        Tuple of (waveform_png, energy_png, comparison_png or None)
    """
    # Calculate times for waveform plot
    fixed_start_time = time_axis[fixed_start_idx]
    fixed_end_time = time_axis[fixed_end_idx]
    
    waveform_png = plot_raw_waveform(
        filtered_audio, sample_rate, fixed_start_time, fixed_end_time
    )
    
    energy_png = plot_energy_curve(
        time_axis, energy, noise_floor, fixed_start_idx, fixed_end_idx
    )
    
    comparison_png = None
    if alt_start_idx is not None and alt_end_idx is not None:
        comparison_png = plot_detection_comparison(
            time_axis, energy, noise_floor,
            fixed_start_idx, fixed_end_idx,
            alt_start_idx, alt_end_idx
        )
    
    return waveform_png, energy_png, comparison_png


def plot_flow_stability(
    time_axis: np.ndarray,
    flow_rate: np.ndarray,
    stable_mask: np.ndarray,
    qmax_standard: float,
    qmax_slope_stabilized: float,
    slope_threshold: float,
    title: str = "Flow Curve: Slope-Stabilized Qmax Detection"
) -> bytes:
    """
    Plot flow curve with stable regions highlighted.
    
    Shows where Qmax can be reliably measured (stable slope regions)
    vs unstable regions (high dQ/dt).
    
    Args:
        time_axis: Time values for each flow frame (trimmed to voiding)
        flow_rate: Flow rate values (ml/s)
        stable_mask: Boolean array - True for stable (low slope) regions
        qmax_standard: Standard Qmax value (without slope restriction)
        qmax_slope_stabilized: Qmax from stable regions only
        slope_threshold: Adaptive slope threshold used
        title: Plot title
        
    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e293b')
    ax.set_facecolor('#0f172a')
    
    # Plot unstable regions as background shading
    unstable_mask = ~stable_mask
    if np.any(unstable_mask):
        # Find contiguous unstable regions
        starts = np.where(np.diff(unstable_mask.astype(int)) == 1)[0] + 1
        ends = np.where(np.diff(unstable_mask.astype(int)) == -1)[0] + 1
        
        # Handle edge cases
        if unstable_mask[0]:
            starts = np.concatenate([[0], starts])
        if unstable_mask[-1]:
            ends = np.concatenate([ends, [len(unstable_mask)]])
        
        for start, end in zip(starts, ends):
            if start < len(time_axis) and end <= len(time_axis):
                ax.axvspan(time_axis[start], time_axis[min(end, len(time_axis)-1)], 
                          alpha=0.25, color='#ef4444', label='_Unstable (high slope)')
    
    # Plot stable regions as background shading
    if np.any(stable_mask):
        starts = np.where(np.diff(stable_mask.astype(int)) == 1)[0] + 1
        ends = np.where(np.diff(stable_mask.astype(int)) == -1)[0] + 1
        
        if stable_mask[0]:
            starts = np.concatenate([[0], starts])
        if stable_mask[-1]:
            ends = np.concatenate([ends, [len(stable_mask)]])
        
        for start, end in zip(starts, ends):
            if start < len(time_axis) and end <= len(time_axis):
                ax.axvspan(time_axis[start], time_axis[min(end, len(time_axis)-1)], 
                          alpha=0.15, color='#22c55e', label='_Stable (low slope)')
    
    # Plot flow curve
    ax.plot(time_axis, flow_rate, color='#38bdf8', linewidth=1.5, label='Flow Rate')
    
    # Horizontal lines for Qmax values
    ax.axhline(y=qmax_standard, color='#f59e0b', linestyle='--', linewidth=2, 
               label=f'Qmax Standard: {qmax_standard:.1f} ml/s')
    ax.axhline(y=qmax_slope_stabilized, color='#22c55e', linestyle='-', linewidth=2,
               label=f'Qmax Slope-Stabilized: {qmax_slope_stabilized:.1f} ml/s')
    
    # Create legend with region indicators
    import matplotlib.patches as mpatches
    stable_patch = mpatches.Patch(color='#22c55e', alpha=0.3, label='Stable Region (|dQ/dt| ≤ threshold)')
    unstable_patch = mpatches.Patch(color='#ef4444', alpha=0.3, label='Unstable Region (|dQ/dt| > threshold)')
    
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([stable_patch, unstable_patch])
    ax.legend(handles=handles, loc='upper right', facecolor='#1e293b', 
              edgecolor='#475569', labelcolor='white', fontsize=8)
    
    # Info box
    stable_pct = np.sum(stable_mask) / len(stable_mask) * 100
    info_text = (f'Slope threshold: {slope_threshold:.2f} ml/s²\n'
                 f'Stable regions: {stable_pct:.0f}% of voiding\n'
                 f'Qmax reduction: {qmax_standard - qmax_slope_stabilized:+.1f} ml/s')
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, color='white', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#334155', edgecolor='#475569', alpha=0.9))
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=10)
    ax.set_ylabel('Flow Rate (ml/s)', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='#475569')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='#1e293b', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()

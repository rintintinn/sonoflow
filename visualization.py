"""
Visualization module for acoustic uroflowmetry.
Generates clinical-style uroflowmetry graphs.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from processor import ProcessingResult


def generate_clinical_graph(result: ProcessingResult, output_path: str = None) -> tuple[bytes, str]:
    """
    Generate a clinical-style uroflowmetry graph.
    
    Args:
        result: ProcessingResult from audio processor
        output_path: Optional path to save PNG (if None, returns bytes)
        
    Returns:
        Tuple of (PNG image as bytes, timestamp string for filename)
    """
    # Set up the figure with clinical styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # Background and grid
    ax.set_facecolor('#fafafa')
    ax.grid(True, linestyle='-', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color='#dddddd', alpha=0.5)
    
    # Plot flow curve with dot markers
    ax.plot(result.time, result.flow_rate, 
            color='#1a5276', linewidth=2, label='Flow Rate')
    ax.scatter(result.time[::5], result.flow_rate[::5], 
               color='#2874a6', s=15, zorder=5, alpha=0.7)
    
    # Fill under curve
    ax.fill_between(result.time, result.flow_rate, 
                    alpha=0.3, color='#3498db')
    
    # Mark both Qmax values for validation
    qmax_smoothed_idx = np.argmax(result.flow_rate)
    
    # Qmax from minimal smoothing (red, dashed line above curve)
    ax.axhline(y=result.qmax, color='#c0392b', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Qmax (min): {result.qmax:.1f} ml/s')
    
    # Qmax from full smoothing (green, on the curve)
    ax.axhline(y=result.qmax_smoothed, color='#27ae60', linestyle=':', 
               linewidth=1, alpha=0.7, label=f'Qmax (smooth): {result.qmax_smoothed:.1f} ml/s')
    ax.scatter([result.time[qmax_smoothed_idx]], [result.qmax_smoothed], 
               color='#27ae60', s=80, zorder=10, marker='o')
    
    # Date and time stamp - centered at top
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_filename = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Axes labels and title with timestamp
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flow Rate (ml/s)', fontsize=12, fontweight='bold')
    ax.set_title(f'Acoustic Uroflowmetry\n{timestamp_str}', fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, max(result.time) * 1.05)
    ax.set_ylim(0, result.qmax * 1.3)
    
    # Minor ticks
    ax.minorticks_on()
    
    # Annotations box with all Qmax values for validation
    annotation_text = (
        f"Qmax (peak): {result.qmax:.1f} ml/s\n"
        f"Qmax (smooth): {result.qmax_smoothed:.1f} ml/s\n"
        f"Qmax (ICS-slide): {result.qmax_ics_sliding:.1f} ml/s\n"
        f"Qmax (ICS-consec): {result.qmax_ics_consecutive:.1f} ml/s\n"
        f"Qavg: {result.qavg:.1f} ml/s\n"
        f"Volume: {result.volume_ml:.0f} ml\n"
        f"Voiding Time: {result.voiding_time:.1f} s"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='#2c3e50', alpha=0.9)
    ax.text(0.98, 0.97, annotation_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')
    
    # (Timestamp now shown in title)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    image_bytes = buf.read()
    
    # Optionally save to file
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
    
    plt.close(fig)
    
    return image_bytes, timestamp_filename

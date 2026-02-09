"""
Spectral Analysis Visualization for Acoustic Uroflowmetry

Generates a multi-panel debug plot showing:
    Panel 1: RMS Energy with all detection method markers
    Panel 2: Spectral Centroid over time
    Panel 3: Band Energy Ratio over time
    Panel 4: Voiding Likelihood Score with threshold

Intended for research comparison — evaluating spectral detection
accuracy against the primary energy-based detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from typing import Optional


def plot_spectral_analysis(
    time_axis: np.ndarray,
    energy: np.ndarray,
    noise_floor: float,
    # Primary detection (Otsu+changepoint multi-episode)
    primary_start_idx: int,
    primary_end_idx: int,
    # Legacy detection (fixed threshold)
    legacy_start_idx: Optional[int] = None,
    legacy_end_idx: Optional[int] = None,
    # Spectral detection results
    spectral_centroid: Optional[np.ndarray] = None,
    band_energy_ratio: Optional[np.ndarray] = None,
    spectral_flatness: Optional[np.ndarray] = None,
    voiding_likelihood: Optional[np.ndarray] = None,
    spectral_start_idx: Optional[int] = None,
    spectral_end_idx: Optional[int] = None,
    likelihood_threshold: float = 0.4,
    title: str = "Spectral Analysis: Voiding Detection Comparison",
) -> bytes:
    """
    Generate multi-panel spectral analysis plot.

    Args:
        time_axis: Time values for each frame
        energy: RMS energy per frame
        noise_floor: Estimated noise floor
        primary_start_idx/end_idx: Current default detection boundaries
        legacy_start_idx/end_idx: Legacy fixed-threshold boundaries (optional)
        spectral_centroid: Per-frame spectral centroid in Hz
        band_energy_ratio: Per-frame BER (0-1)
        spectral_flatness: Per-frame spectral flatness (0-1)
        voiding_likelihood: Per-frame voiding likelihood (0-1)
        spectral_start_idx/end_idx: Spectral detection boundaries
        likelihood_threshold: Threshold used for spectral detection
        title: Overall plot title

    Returns:
        PNG image as bytes
    """
    # Determine number of panels
    has_spectral = voiding_likelihood is not None
    n_panels = 4 if has_spectral else 1

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 3.5 * n_panels),
        facecolor="#1e293b",
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    if n_panels == 1:
        axes = [axes]

    # =====================================================================
    # Colour palette
    # =====================================================================
    C_ENERGY = "#38bdf8"      # Light blue
    C_PRIMARY = "#22c55e"     # Green — primary (Otsu+CP)
    C_LEGACY = "#a855f7"      # Purple — legacy fixed-threshold
    C_SPECTRAL = "#f59e0b"    # Amber — spectral detection
    C_CENTROID = "#f472b6"    # Pink
    C_BER = "#34d399"         # Emerald
    C_FLATNESS = "#c084fc"    # Light purple
    C_LIKELIHOOD = "#fb923c"  # Orange
    C_NOISE = "#f59e0b"       # Amber (noise floor)

    # Helper: draw detection boundaries on an axis
    def draw_boundaries(ax, show_legend_box=False):
        """Draw vertical lines for all detection methods."""
        primary_start_t = time_axis[primary_start_idx]
        primary_end_t = time_axis[primary_end_idx]
        ax.axvline(primary_start_t, color=C_PRIMARY, ls="-", lw=1.8, alpha=0.9)
        ax.axvline(primary_end_t, color=C_PRIMARY, ls="-", lw=1.8, alpha=0.9)

        if legacy_start_idx is not None and legacy_end_idx is not None:
            legacy_start_t = time_axis[legacy_start_idx]
            legacy_end_t = time_axis[legacy_end_idx]
            ax.axvline(legacy_start_t, color=C_LEGACY, ls="--", lw=1.5, alpha=0.8)
            ax.axvline(legacy_end_t, color=C_LEGACY, ls="--", lw=1.5, alpha=0.8)

        if has_spectral and spectral_start_idx is not None and spectral_end_idx is not None:
            spec_start_t = time_axis[spectral_start_idx]
            spec_end_t = time_axis[spectral_end_idx]
            ax.axvline(spec_start_t, color=C_SPECTRAL, ls="-.", lw=1.8, alpha=0.9)
            ax.axvline(spec_end_t, color=C_SPECTRAL, ls="-.", lw=1.8, alpha=0.9)

    def style_axis(ax, ylabel):
        """Apply consistent dark styling."""
        ax.set_facecolor("#0f172a")
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="#475569")

    # =====================================================================
    # Panel 1: RMS Energy with all detection markers
    # =====================================================================
    ax1 = axes[0]
    ax1.plot(time_axis, energy, color=C_ENERGY, lw=1.2, alpha=0.85, label="RMS Energy")
    ax1.axhline(noise_floor, color=C_NOISE, ls="-", lw=1, alpha=0.6, label="Noise floor")
    draw_boundaries(ax1, show_legend_box=True)
    style_axis(ax1, "RMS Energy")
    ax1.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)

    # Build legend
    legend_handles = [
        mpatches.Patch(color=C_PRIMARY, label=f"Primary (Otsu+CP): {time_axis[primary_start_idx]:.1f}s – {time_axis[primary_end_idx]:.1f}s"),
    ]
    if legacy_start_idx is not None:
        legend_handles.append(
            mpatches.Patch(color=C_LEGACY, label=f"Legacy (fixed): {time_axis[legacy_start_idx]:.1f}s – {time_axis[legacy_end_idx]:.1f}s")
        )
    if has_spectral and spectral_start_idx is not None:
        legend_handles.append(
            mpatches.Patch(color=C_SPECTRAL, label=f"Spectral: {time_axis[spectral_start_idx]:.1f}s – {time_axis[spectral_end_idx]:.1f}s")
        )
    ax1.legend(
        handles=legend_handles, loc="upper right",
        facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8,
    )

    # Timing summary box
    primary_dur = time_axis[primary_end_idx] - time_axis[primary_start_idx]
    info_parts = [f"Primary: {primary_dur:.1f}s"]
    if legacy_start_idx is not None:
        legacy_dur = time_axis[legacy_end_idx] - time_axis[legacy_start_idx]
        info_parts.append(f"Legacy: {legacy_dur:.1f}s")
    if has_spectral and spectral_start_idx is not None:
        spec_dur = time_axis[spectral_end_idx] - time_axis[spectral_start_idx]
        info_parts.append(f"Spectral: {spec_dur:.1f}s")
    info_text = " | ".join(info_parts)
    ax1.text(
        0.02, 0.92, info_text, transform=ax1.transAxes, color="white", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="#334155", edgecolor="#475569", alpha=0.9),
    )

    if not has_spectral:
        ax1.set_xlabel("Time (seconds)", color="white", fontsize=10)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor="#1e293b", edgecolor="none")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    # =====================================================================
    # Panel 2: Spectral Centroid
    # =====================================================================
    ax2 = axes[1]
    ax2.plot(time_axis, spectral_centroid, color=C_CENTROID, lw=1.0, alpha=0.8)
    # Shade voiding centroid range
    ax2.axhspan(600, 2200, alpha=0.08, color=C_CENTROID, label="Voiding range (600-2200 Hz)")
    ax2.axhline(1200, color=C_CENTROID, ls=":", lw=0.8, alpha=0.4, label="Peak (1200 Hz)")
    draw_boundaries(ax2)
    style_axis(ax2, "Spectral Centroid (Hz)")
    ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569",
               labelcolor="white", fontsize=7)

    # =====================================================================
    # Panel 3: Band Energy Ratio
    # =====================================================================
    ax3 = axes[2]
    ax3.plot(time_axis, band_energy_ratio, color=C_BER, lw=1.0, alpha=0.8)
    ax3.axhline(0.35, color=C_BER, ls=":", lw=0.8, alpha=0.5, label="Sigmoid centre (0.35)")
    # Optionally overlay spectral flatness on secondary axis
    if spectral_flatness is not None:
        ax3b = ax3.twinx()
        ax3b.plot(time_axis, spectral_flatness, color=C_FLATNESS, lw=0.8, alpha=0.5)
        ax3b.set_ylabel("Spectral Flatness", color=C_FLATNESS, fontsize=8)
        ax3b.tick_params(axis="y", colors=C_FLATNESS, labelsize=7)
        ax3b.spines["right"].set_color(C_FLATNESS)
        ax3b.set_ylim(0, 1)
    draw_boundaries(ax3)
    style_axis(ax3, "Band Energy Ratio")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569",
               labelcolor="white", fontsize=7)

    # =====================================================================
    # Panel 4: Voiding Likelihood Score
    # =====================================================================
    ax4 = axes[3]
    ax4.fill_between(time_axis, voiding_likelihood, alpha=0.3, color=C_LIKELIHOOD)
    ax4.plot(time_axis, voiding_likelihood, color=C_LIKELIHOOD, lw=1.2, alpha=0.9,
             label="Voiding Likelihood")
    ax4.axhline(likelihood_threshold, color="#ef4444", ls="--", lw=1.2, alpha=0.7,
                label=f"Threshold ({likelihood_threshold:.2f})")
    draw_boundaries(ax4)
    style_axis(ax4, "Voiding Likelihood")
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Time (seconds)", color="white", fontsize=10)
    ax4.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569",
               labelcolor="white", fontsize=8)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#1e293b", edgecolor="none")
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()

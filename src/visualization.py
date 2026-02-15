"""Visualization: note histogram and pitch contour charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from src.analysis import VocalRangeResult, midi_to_note_name


# Standard voice type ranges as (name, low_midi, high_midi, color)
VOICE_TYPES = [
    ("Bass",     40, 64, "#1976D2"),    # E2-E4
    ("Baritone", 45, 69, "#388E3C"),    # A2-A4
    ("Tenor",    48, 72, "#F57C00"),    # C3-C5
    ("Alto",     53, 77, "#7B1FA2"),    # F3-F5
    ("Mezzo",    57, 81, "#C2185B"),    # A3-A5
    ("Soprano",  60, 84, "#D32F2F"),    # C4-C6
]

_VOICE_TYPE_ABBREVS = {
    "Bass": "Bass", "Baritone": "Bar.", "Tenor": "Ten.",
    "Alto": "Alto", "Mezzo": "Mez.", "Soprano": "Sop.",
}

_BAR_CMAP = LinearSegmentedColormap.from_list(
    "vocal_bars", ["#1565C0", "#7E57C2", "#EF6C00"],
)


def _note_sort_key(name: str) -> int:
    """Parse note name like 'C4' to MIDI number for sorting."""
    from src.analysis import NOTE_NAMES
    for i, n in enumerate(NOTE_NAMES):
        if name.startswith(n) and (len(n) == len(name.rstrip("0123456789-"))):
            octave = int(name[len(n):])
            return (octave + 1) * 12 + i
    return 0


def plot_vocal_range(
    result: VocalRangeResult,
    title: str = "Vocal Range Analysis",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Create horizontal bar chart with a voice-type range indicator panel."""
    if not result.note_histogram:
        raise ValueError("No notes to plot — histogram is empty.")

    sorted_notes = sorted(result.note_histogram.items(), key=lambda x: _note_sort_key(x[0]))
    note_names = [n for n, _ in sorted_notes]
    durations = [d for _, d in sorted_notes]
    n_notes = len(note_names)

    # Bar colors: pitch-based gradient (blue → teal → amber)
    bar_colors = [_BAR_CMAP(i / max(1, n_notes - 1)) for i in range(n_notes)]

    # Ensure tiny bars are still visible (min 1.5% of max)
    max_dur = max(durations)
    min_visible = max_dur * 0.015
    display_durations = [max(d, min_visible) for d in durations]

    # Layout: main chart + narrow range indicator panel
    fig_height = max(4, n_notes * 0.35 + 2)
    fig = plt.figure(figsize=(11, fig_height))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(1, 2, width_ratios=[8, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_range = fig.add_subplot(gs[1], sharey=ax)

    ax.set_facecolor("#FAFAFA")
    ax_range.set_facecolor("white")

    # === Main bar chart ===
    ax.barh(
        range(n_notes), display_durations,
        color=bar_colors, edgecolor="white", linewidth=0.5,
        height=0.7, zorder=2,
    )

    ax.set_yticks(range(n_notes))
    ax.set_yticklabels(note_names, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Duration (seconds)", fontsize=10, color="#555")
    ax.set_xlim(0, max_dur * 1.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDD")
    ax.spines["bottom"].set_color("#DDD")
    ax.xaxis.grid(True, alpha=0.15, linewidth=0.5, color="#999")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555", which="both")

    # Title and subtitle
    subtitle = f"Range: {result.lowest_note} \u2013 {result.highest_note}  \u00b7  {result.range_display}"
    ax.set_title(subtitle, fontsize=10, color="#888", pad=20)
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222")

    # === Voice type range indicator panel (right) ===
    n_types = len(VOICE_TYPES)
    for idx, (vtype_name, low_midi, high_midi, color) in enumerate(VOICE_TYPES):
        y_positions = [
            i for i, n in enumerate(note_names)
            if low_midi <= _note_sort_key(n) <= high_midi
        ]
        if y_positions:
            ax_range.barh(
                y_positions, [0.85] * len(y_positions),
                left=idx, color=color, alpha=0.75,
                height=0.95, edgecolor="white", linewidth=0.3,
            )

    # Column headers (voice type abbreviations)
    for idx, (vtype_name, _, _, color) in enumerate(VOICE_TYPES):
        ax_range.text(
            (idx + 0.42) / n_types, 1.01,
            _VOICE_TYPE_ABBREVS.get(vtype_name, vtype_name[:3]),
            rotation=90, ha="center", va="bottom",
            transform=ax_range.transAxes,
            fontsize=6.5, color=color, fontweight="bold",
        )

    ax_range.set_xlim(0, n_types)
    ax_range.set_xticks([])
    for spine in ax_range.spines.values():
        spine.set_visible(False)
    ax_range.tick_params(left=False, labelleft=False)

    gs.tight_layout(fig, rect=[0, 0, 1, 0.96])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)


def plot_pitch_contour(
    result: VocalRangeResult,
    title: str = "Pitch Contour",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot raw F0 contour over time."""
    voiced = [f for f in result.pitch_frames if f.frequency_hz > 0 and f.confidence > 0]
    if not voiced:
        raise ValueError("No voiced frames to plot.")

    times = [f.time_seconds for f in voiced]
    freqs = [f.frequency_hz for f in voiced]
    confs = [f.confidence for f in voiced]

    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(times, freqs, c=confs, s=1, cmap="viridis", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add note name gridlines
    reference_notes = [("C2", 65.41), ("E2", 82.41), ("A2", 110.0),
                       ("C3", 130.81), ("E3", 164.81), ("A3", 220.0),
                       ("C4", 261.63), ("E4", 329.63), ("A4", 440.0),
                       ("C5", 523.25), ("E5", 659.25), ("A5", 880.0),
                       ("C6", 1046.5)]
    for note_name, freq in reference_notes:
        if min(freqs) * 0.8 <= freq <= max(freqs) * 1.2:
            ax.axhline(y=freq, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
            ax.text(max(times) * 1.01, freq, note_name, fontsize=7, va="center", color="gray")

    plt.colorbar(scatter, ax=ax, label="Confidence", shrink=0.8)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

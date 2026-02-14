"""Compare all pitch detector results."""
from common import RESULTS_DIR, EXPECTED_TOP_NOTES, EXPECTED_OTHER_NOTES, Frame, enforce_memory_limit

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analysis import PitchFrame, build_note_histogram, hz_to_note_name

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DETECTORS = ["crepe", "basicpitch", "swiftf0", "fcpe", "penn", "pesto", "rmvpe"]


def load_all() -> dict[str, dict]:
    """Load all available result files."""
    results = {}
    for name in DETECTORS:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def summary_table(results: dict[str, dict]):
    """Print a comparison summary table."""
    print(f"\n{'Detector':<15} {'Time (s)':>8} {'Peak MB':>8} {'Frames':>8} {'Voiced':>8} {'Top 3 Notes':<30} {'Match'}")
    print("-" * 95)

    for name, data in results.items():
        frames = [Frame(f["time"], f["freq"], f["confidence"]) for f in data["frames"]]
        pitch_frames = [PitchFrame(f.time, f.freq, f.confidence) for f in frames]
        histogram = build_note_histogram(pitch_frames, confidence_threshold=0.3)
        top3 = sorted(histogram.items(), key=lambda x: -x[1])[:3]
        top3_names = {n for n, _ in top3}
        top3_str = ", ".join(f"{n}({d:.1f}s)" for n, d in top3)
        voiced = sum(1 for f in frames if f.freq > 0)

        overlap = top3_names & EXPECTED_TOP_NOTES
        match = f"{len(overlap)}/3"

        peak_mb = data.get("peak_memory_gb", 0) * 1024
        mem_str = f"{peak_mb:.0f}" if peak_mb > 0 else "n/a"

        print(f"{name:<15} {data['elapsed_seconds']:>8.2f} {mem_str:>8} {len(frames):>8} {voiced:>8} {top3_str:<30} {match}")


def consensus_analysis(results: dict[str, dict]):
    """Per-frame majority vote across detectors; measure each detector's agreement."""
    # Build time->note lookup per detector
    note_maps = {}
    for name, data in results.items():
        nmap = {}
        for f in data["frames"]:
            if f["freq"] > 0 and f["confidence"] > 0.3:
                t = round(f["time"], 2)
                nmap[t] = hz_to_note_name(f["freq"])
        note_maps[name] = nmap

    # Collect all time points
    all_times = set()
    for nmap in note_maps.values():
        all_times.update(nmap.keys())
    times = sorted(all_times)

    agreement_counts = {name: 0 for name in results}
    total_voted = 0

    for t in times:
        votes = {}
        for name in results:
            note = note_maps[name].get(t)
            if note:
                votes[note] = votes.get(note, 0) + 1
        if not votes:
            continue

        majority_note = max(votes, key=votes.get)
        if votes[majority_note] < 2:
            continue

        total_voted += 1
        for name in results:
            if note_maps[name].get(t) == majority_note:
                agreement_counts[name] += 1

    print(f"\n{'=' * 55}")
    print(f"  Consensus Agreement ({total_voted} frames with majority)")
    print(f"{'=' * 55}")
    for name, count in sorted(agreement_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_voted * 100) if total_voted > 0 else 0
        print(f"  {name:<15} {count:>6}/{total_voted} ({pct:.1f}%)")


def plot_contours(results: dict[str, dict]):
    """Overlay pitch contours for all detectors."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        times, freqs = [], []
        for f in data["frames"]:
            if f["freq"] > 0 and f["confidence"] > 0.3:
                times.append(f["time"])
                freqs.append(f["freq"])
        ax.scatter(times, freqs, s=0.5, alpha=0.5, color=color, label=name)

    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Pitch Contour Comparison")
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    out = RESULTS_DIR / "contour_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved contour comparison to {out}")
    plt.close(fig)


def plot_histograms(results: dict[str, dict]):
    """Side-by-side note histograms."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        frames = [PitchFrame(f["time"], f["freq"], f["confidence"]) for f in data["frames"]]
        histogram = build_note_histogram(frames, confidence_threshold=0.3)
        top = sorted(histogram.items(), key=lambda x: -x[1])[:10]
        if not top:
            ax.set_title(name)
            continue
        notes, durations = zip(*top)
        colors = ["#4CAF50" if n in EXPECTED_TOP_NOTES
                  else "#FF9800" if n in EXPECTED_OTHER_NOTES
                  else "#2196F3" for n in notes]
        ax.barh(range(len(notes)), durations, color=colors)
        ax.set_yticks(range(len(notes)))
        ax.set_yticklabels(notes)
        ax.set_title(f"{name}\n({data['elapsed_seconds']:.1f}s)")
        ax.invert_yaxis()

    fig.suptitle("Note Histograms (green=expected primary, orange=expected secondary)")
    fig.tight_layout()
    out = RESULTS_DIR / "histogram_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved histogram comparison to {out}")
    plt.close(fig)


def main():
    enforce_memory_limit(16)
    results = load_all()
    if not results:
        print("No results found. Run the individual detector scripts first.")
        return

    print(f"Found results for: {', '.join(results.keys())}")
    summary_table(results)
    consensus_analysis(results)
    plot_contours(results)
    plot_histograms(results)


if __name__ == "__main__":
    main()

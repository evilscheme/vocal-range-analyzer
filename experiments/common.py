"""Shared utilities for pitch-detection experiments."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable so we can use src.analysis
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUDIO_FILE = Path.home() / "Downloads" / "The Cuckoo (Remastered 2025) [ns1_1jU4Ycw].mp4"
RESULTS_DIR = Path(__file__).parent / "results"
VOCALS_PATH = RESULTS_DIR / "vocals.wav"

EXPECTED_TOP_NOTES: set[str] = {"C4", "E4", "A3"}
EXPECTED_OTHER_NOTES: set[str] = {"D4", "E3", "G3"}


# ---------------------------------------------------------------------------
# Frame dataclass
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """A single pitch observation from a detector."""
    time: float
    freq: float
    confidence: float


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------
def _get_rss_gb() -> float:
    """Return current process RSS in gigabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: resource.getrusage on macOS (ru_maxrss is in bytes on macOS)
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 ** 3)
    else:
        return usage.ru_maxrss / (1024 ** 2)


def enforce_memory_limit(max_gb: float = 16) -> None:
    """Start a daemon thread that monitors RSS and exits if it exceeds *max_gb*.

    Call this at the top of every experiment run script.
    """
    def _watchdog() -> None:
        while True:
            rss = _get_rss_gb()
            if rss > max_gb:
                print(
                    f"\n[MEMORY GUARD] RSS is {rss:.2f} GB, exceeding limit of "
                    f"{max_gb} GB. Terminating process to protect the system.",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(1)
            time.sleep(2)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------
@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that measures wall-clock elapsed time.

    Usage::

        with timer() as t:
            do_work()
        print(t["elapsed"])
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Load an audio file as mono float32 numpy array via librosa."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def load_vocals(sr: int = 16000) -> np.ndarray:
    """Load the cached separated vocals from *VOCALS_PATH*.

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    if not VOCALS_PATH.exists():
        raise FileNotFoundError(
            f"Vocals file not found at {VOCALS_PATH}. "
            "Run the vocal separation step first to generate it."
        )
    return load_audio(VOCALS_PATH, sr=sr)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_results(
    name: str,
    frames: list[Frame],
    elapsed: float,
    settings: dict[str, Any] | None = None,
) -> Path:
    """Save experiment results to ``RESULTS_DIR/{name}.json``.

    Returns the path to the saved JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "detector": name,
        "elapsed_seconds": elapsed,
        "num_frames": len(frames),
        "settings": settings or {},
        "frames": [asdict(f) for f in frames],
    }
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_results(name: str) -> dict[str, Any]:
    """Load a previously saved result JSON by detector *name*."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results found for '{name}' at {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(name: str, frames: list[Frame], elapsed: float) -> None:
    """Print a human-readable summary of detection results.

    Shows timing, frame counts, and the top 6 notes with ground-truth
    match indicators (using ``src.analysis.PitchFrame`` and
    ``build_note_histogram`` from the main codebase).
    """
    from src.analysis import PitchFrame, build_note_histogram

    voiced = [f for f in frames if f.freq > 0 and f.confidence > 0.5]

    # Convert to PitchFrame for the histogram builder
    pitch_frames = [
        PitchFrame(
            time_seconds=f.time,
            frequency_hz=f.freq,
            confidence=f.confidence,
        )
        for f in frames
    ]

    histogram = build_note_histogram(pitch_frames)

    # Sort notes by total duration descending
    sorted_notes = sorted(histogram.items(), key=lambda kv: kv[1], reverse=True)
    top_notes = sorted_notes[:6]

    all_expected = EXPECTED_TOP_NOTES | EXPECTED_OTHER_NOTES

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Elapsed      : {elapsed:.2f} s")
    print(f"  Total frames : {len(frames)}")
    print(f"  Voiced frames: {len(voiced)}")
    print(f"  Top notes:")
    for note, duration in top_notes:
        if note in EXPECTED_TOP_NOTES:
            tag = " << TOP"
        elif note in EXPECTED_OTHER_NOTES:
            tag = " < expected"
        else:
            tag = ""
        print(f"    {note:>4s}  {duration:6.2f}s{tag}")
    print(f"{'=' * 50}\n")

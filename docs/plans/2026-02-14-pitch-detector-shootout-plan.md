# Pitch Detector Shootout — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run 7 pitch detectors on "The Cuckoo" and compare accuracy + speed to find the best CREPE replacement.

**Architecture:** Throwaway experiment scripts in `experiments/`. Each detector gets its own `run_*.py` that outputs standardized JSON. A `compare.py` script reads all results and produces a summary table + plots. Demucs separation is run once and cached.

**Tech Stack:** Python, librosa, torch, matplotlib. New deps: swift-f0, torchfcpe, penn, pesto-pitch. RMVPE cloned from GitHub.

**Reference file:** `/Users/bryan/Downloads/The Cuckoo (Remastered 2025) [ns1_1jU4Ycw].mp4`

**Known ground truth:** Melody notes C4, E4, A3 (primary); D4, E3, G3 (secondary).

---

### Task 1: Scaffold experiments directory and common utilities

**Files:**
- Create: `experiments/common.py`
- Create: `experiments/.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p experiments/results
```

**Step 2: Create .gitignore for results**

Create `experiments/.gitignore`:
```
results/
```

**Step 3: Write common.py**

This module provides shared utilities for all run scripts: audio loading, result serialization, timing, and summary printing.

```python
"""Shared utilities for pitch detector experiments."""
from __future__ import annotations

import json
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import librosa

AUDIO_FILE = Path.home() / "Downloads" / "The Cuckoo (Remastered 2025) [ns1_1jU4Ycw].mp4"
RESULTS_DIR = Path(__file__).parent / "results"
VOCALS_PATH = RESULTS_DIR / "vocals.wav"

# Known ground truth for sanity checking
EXPECTED_TOP_NOTES = {"C4", "E4", "A3"}
EXPECTED_OTHER_NOTES = {"D4", "E3", "G3"}

# Add project root to path so we can import src.analysis
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Frame:
    time: float
    freq: float
    confidence: float


@contextmanager
def timer():
    """Context manager that yields a dict; sets 'elapsed' on exit."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - start


def load_audio(path: Path | str = AUDIO_FILE, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file as mono float32 numpy array."""
    audio, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    return audio, sample_rate


def load_vocals(sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load cached separated vocals."""
    if not VOCALS_PATH.exists():
        raise FileNotFoundError(
            f"Vocals not cached at {VOCALS_PATH}. "
            "Run Demucs first: python experiments/separate_vocals.py"
        )
    audio, sample_rate = librosa.load(str(VOCALS_PATH), sr=sr, mono=True)
    return audio, sample_rate


def save_results(name: str, frames: list[Frame], elapsed: float, settings: dict | None = None):
    """Save results to JSON in experiments/results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    data = {
        "detector": name,
        "elapsed_seconds": round(elapsed, 3),
        "num_frames": len(frames),
        "settings": settings or {},
        "frames": [asdict(f) for f in frames],
    }
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(frames)} frames to {out_path}")


def load_results(name: str) -> dict:
    """Load results JSON for a detector."""
    path = RESULTS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def print_summary(name: str, frames: list[Frame], elapsed: float):
    """Print a quick summary: timing, frame count, top notes."""
    from src.analysis import PitchFrame, build_note_histogram

    pitch_frames = [PitchFrame(f.time, f.freq, f.confidence) for f in frames]
    histogram = build_note_histogram(pitch_frames, confidence_threshold=0.3)
    top_notes = sorted(histogram.items(), key=lambda x: -x[1])[:6]

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Frames:  {len(frames)} total, {sum(1 for f in frames if f.freq > 0)} voiced")
    print(f"\n  Top notes:")
    for note, dur in top_notes:
        bar = "#" * int(dur * 5)
        match = ""
        if note in EXPECTED_TOP_NOTES:
            match = " <-- expected primary"
        elif note in EXPECTED_OTHER_NOTES:
            match = " <-- expected secondary"
        print(f"    {note:>4}: {dur:.2f}s {bar}{match}")
```

**Step 4: Commit**

```bash
git add experiments/
git commit -m "feat: scaffold experiments directory with common utilities"
```

---

### Task 2: Cache separated vocals with Demucs

**Files:**
- Create: `experiments/separate_vocals.py`

**Step 1: Write the separation script**

This reuses the existing `src/separation.py` module to run Demucs once and cache the result.

```python
"""Run Demucs separation once and cache vocals WAV."""
from common import AUDIO_FILE, RESULTS_DIR, timer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.separation import isolate_vocals


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Separating vocals from: {AUDIO_FILE}")

    with timer() as t:
        vocals, sr = isolate_vocals(AUDIO_FILE, output_dir=RESULTS_DIR)

    print(f"Done in {t['elapsed']:.1f}s")
    print(f"Vocals saved to: {RESULTS_DIR / 'vocals.wav'}")
    print(f"Shape: {vocals.shape}, SR: {sr}")


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
cd experiments && python separate_vocals.py
```

Expected: Creates `experiments/results/vocals.wav`. Should take 30-60s.

**Step 3: Commit**

```bash
git add experiments/separate_vocals.py
git commit -m "feat: add Demucs vocal separation caching script"
```

---

### Task 3: run_crepe.py (baseline)

**Files:**
- Create: `experiments/run_crepe.py`

**Step 1: Write the script**

```python
"""Baseline: torchcrepe pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer


def main():
    import torch
    import torchcrepe

    audio, sr = load_vocals(sr=16000)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    hop_length = 160  # 10ms at 16kHz

    with timer() as t:
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                audio_tensor, sr,
                hop_length=hop_length, fmin=65.0, fmax=2000.0,
                model="full", return_periodicity=True,
                batch_size=2048, device=device,
            )
        periodicity = torchcrepe.filter.median(periodicity, win_length=3)
        pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)

    pitch_np = pitch.squeeze().cpu().numpy()
    periodicity_np = periodicity.squeeze().cpu().numpy()
    hop_sec = hop_length / sr

    frames = []
    for i, (f, p) in enumerate(zip(pitch_np, periodicity_np)):
        import numpy as np
        if np.isnan(f) or f <= 0:
            frames.append(Frame(i * hop_sec, 0.0, 0.0))
        else:
            frames.append(Frame(i * hop_sec, float(f), float(p)))

    save_results("crepe", frames, t["elapsed"], {"model": "full", "hop_length": hop_length})
    print_summary("CREPE (baseline)", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
cd experiments && python run_crepe.py
```

Expected: Prints summary with top notes, saves `results/crepe.json`.

**Step 3: Commit**

```bash
git add experiments/run_crepe.py
git commit -m "feat: add CREPE baseline experiment script"
```

---

### Task 4: run_basicpitch.py (baseline)

**Files:**
- Create: `experiments/run_basicpitch.py`

**Step 1: Write the script**

BasicPitch takes a file path (not numpy array) and works on the full mix (no separation needed).

```python
"""Baseline: BasicPitch pitch detection on full mix."""
from common import AUDIO_FILE, save_results, print_summary, Frame, timer


def main():
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    with timer() as t:
        _, _, note_events = predict(
            str(AUDIO_FILE),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=50.0,
            minimum_frequency=65.0,
            maximum_frequency=2000.0,
        )

    HOP = 0.01
    frames = []
    for start, end, midi_pitch, amplitude, _bends in note_events:
        freq = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
        t_pos = start
        while t_pos < end:
            frames.append(Frame(t_pos, freq, float(amplitude)))
            t_pos += HOP

    # Deduplicate overlapping notes (keep highest confidence per time slot)
    frames.sort(key=lambda f: (f.time, -f.confidence))
    deduped = []
    seen = set()
    for f in frames:
        key = round(f.time, 3)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    frames = deduped

    save_results("basicpitch", frames, t["elapsed"])
    print_summary("BasicPitch (baseline)", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
cd experiments && python run_basicpitch.py
```

**Step 3: Commit**

```bash
git add experiments/run_basicpitch.py
git commit -m "feat: add BasicPitch baseline experiment script"
```

---

### Task 5: Install swift-f0 and write run_swiftf0.py

**Files:**
- Create: `experiments/run_swiftf0.py`

**Step 1: Install**

```bash
pip install swift-f0
```

**Step 2: Write the script**

```python
"""SwiftF0 pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer


def main():
    from swift_f0 import SwiftF0

    audio, sr = load_vocals(sr=16000)
    detector = SwiftF0(confidence_threshold=0.5, fmin=65.0, fmax=2000.0)

    with timer() as t:
        result = detector.detect_from_array(audio, sr)

    frames = [
        Frame(float(ts), float(p) if v else 0.0, float(c))
        for ts, p, c, v in zip(
            result.timestamps, result.pitch_hz, result.confidence, result.voicing
        )
    ]

    save_results("swiftf0", frames, t["elapsed"],
                 {"confidence_threshold": 0.5, "fmin": 65.0, "fmax": 2000.0})
    print_summary("SwiftF0", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Step 3: Run it**

```bash
cd experiments && python run_swiftf0.py
```

**Step 4: Commit**

```bash
git add experiments/run_swiftf0.py
git commit -m "feat: add SwiftF0 experiment script"
```

---

### Task 6: Install torchfcpe and write run_fcpe.py

**Files:**
- Create: `experiments/run_fcpe.py`

**Step 1: Install**

```bash
pip install torchfcpe
```

**Step 2: Write the script**

```python
"""FCPE pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer


def main():
    import torch
    from torchfcpe import spawn_bundled_infer_model

    audio, sr = load_vocals(sr=16000)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    model = spawn_bundled_infer_model(device=device)

    # FCPE expects (batch, samples, 1)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)
    hop = 160  # 10ms at 16kHz
    target_len = (len(audio) // hop) + 1

    with timer() as t:
        f0 = model.infer(
            audio_tensor, sr=sr,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=65, f0_max=2000,
            interp_uv=False,
            output_interp_target_length=target_len,
        )

    f0_np = f0.squeeze().cpu().numpy()
    hop_sec = hop / sr

    frames = [
        Frame(i * hop_sec, float(f) if f > 0 else 0.0, 1.0 if f > 0 else 0.0)
        for i, f in enumerate(f0_np)
    ]

    save_results("fcpe", frames, t["elapsed"],
                 {"decoder_mode": "local_argmax", "threshold": 0.006})
    print_summary("FCPE", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Note:** FCPE doesn't output per-frame confidence directly — it outputs F0 with 0 for unvoiced. We use binary confidence (1.0 if voiced, 0.0 if not). This is a known limitation for this comparison.

**Step 3: Run it**

```bash
cd experiments && python run_fcpe.py
```

**Step 4: Commit**

```bash
git add experiments/run_fcpe.py
git commit -m "feat: add FCPE experiment script"
```

---

### Task 7: Install penn and write run_penn.py

**Files:**
- Create: `experiments/run_penn.py`

**Step 1: Install**

```bash
pip install penn
```

**Step 2: Write the script**

```python
"""PENN (FCNF0++) pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer


def main():
    import torch
    import penn

    audio, sr = load_vocals(sr=16000)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    gpu = None  # CPU
    if torch.cuda.is_available():
        gpu = 0

    with timer() as t:
        pitch, periodicity = penn.from_audio(
            audio_tensor, sr,
            hopsize=0.01,
            fmin=65.0, fmax=2000.0,
            batch_size=2048,
            gpu=gpu,
        )

    pitch_np = pitch.squeeze().cpu().numpy()
    period_np = periodicity.squeeze().cpu().numpy()

    frames = [
        Frame(i * 0.01, float(p), float(c))
        for i, (p, c) in enumerate(zip(pitch_np, period_np))
    ]

    save_results("penn", frames, t["elapsed"],
                 {"hopsize": 0.01, "fmin": 65.0, "fmax": 2000.0})
    print_summary("PENN (FCNF0++)", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Note:** PENN uses `gpu=0` for CUDA, `gpu=None` for CPU. It does not support MPS.

**Step 3: Run it**

```bash
cd experiments && python run_penn.py
```

**Step 4: Commit**

```bash
git add experiments/run_penn.py
git commit -m "feat: add PENN experiment script"
```

---

### Task 8: Install pesto-pitch and write run_pesto.py

**Files:**
- Create: `experiments/run_pesto.py`

**Step 1: Install**

```bash
pip install pesto-pitch
```

**Step 2: Write the script**

```python
"""PESTO pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer


def main():
    import torch
    import pesto

    audio, sr = load_vocals(sr=16000)
    audio_tensor = torch.from_numpy(audio).float()  # 1D: (num_samples,)

    with timer() as t:
        timesteps, pitch, confidence, _ = pesto.predict(
            audio_tensor, sr,
            step_size=10.0,
            convert_to_freq=True,
        )

    ts_np = timesteps.cpu().numpy()
    pitch_np = pitch.cpu().numpy()
    conf_np = confidence.cpu().numpy()

    frames = [
        Frame(float(ts), float(p) if p > 0 else 0.0, float(c))
        for ts, p, c in zip(ts_np, pitch_np, conf_np)
    ]

    save_results("pesto", frames, t["elapsed"], {"step_size_ms": 10.0})
    print_summary("PESTO", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Step 3: Run it**

```bash
cd experiments && python run_pesto.py
```

**Step 4: Commit**

```bash
git add experiments/run_pesto.py
git commit -m "feat: add PESTO experiment script"
```

---

### Task 9: Set up RMVPE and write run_rmvpe.py

**Files:**
- Create: `experiments/run_rmvpe.py`

This is the most complex setup because RMVPE has no pip package. We need to:
1. Download the pretrained weights (`rmvpe.pt`) from Hugging Face
2. Get the `rmvpe.py` inference module from the RVC ecosystem
3. Write our run script

**Step 1: Download model weights**

```bash
mkdir -p experiments/rmvpe_model
curl -L -o experiments/rmvpe_model/rmvpe.pt \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
```

Add to `.gitignore`:
```
rmvpe_model/
```

**Step 2: Get the RMVPE inference class**

Clone the minimal inference code. The RVC ecosystem's `rmvpe.py` is the most battle-tested standalone version. Download it from the Mangio RVC fork or similar source:

```bash
curl -L -o experiments/rmvpe_model/rmvpe.py \
  "https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main/lib/rmvpe.py"
```

If the download URL doesn't work, check the repo structure and adjust. The key class we need is `RMVPE` with an `infer_from_audio(audio, thred)` method.

**Step 3: Write the run script**

```python
"""RMVPE pitch detection on full mix (no separation needed)."""
from common import load_audio, save_results, print_summary, Frame, timer

import sys
from pathlib import Path

# Add rmvpe_model to path so we can import the RMVPE class
sys.path.insert(0, str(Path(__file__).parent / "rmvpe_model"))


def main():
    import torch

    # Import may vary depending on which rmvpe.py source we got
    from rmvpe import RMVPE

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # RMVPE with is_half=True requires CUDA; use False for CPU/MPS

    model_path = str(Path(__file__).parent / "rmvpe_model" / "rmvpe.pt")
    model = RMVPE(model_path, is_half=False, device=device)

    # RMVPE works on full mix — no separation needed
    audio, sr = load_audio(sr=16000)

    with timer() as t:
        f0 = model.infer_from_audio(audio, thred=0.03)

    hop_sec = 160 / 16000  # 10ms
    frames = [
        Frame(i * hop_sec, float(f) if f > 0 else 0.0, 1.0 if f > 0 else 0.0)
        for i, f in enumerate(f0)
    ]

    save_results("rmvpe", frames, t["elapsed"],
                 {"threshold": 0.03, "uses_separation": False})
    print_summary("RMVPE (no separation)", frames, t["elapsed"])


if __name__ == "__main__":
    main()
```

**Step 4: Run it**

```bash
cd experiments && python run_rmvpe.py
```

**Step 5: Commit**

```bash
git add experiments/run_rmvpe.py experiments/.gitignore
git commit -m "feat: add RMVPE experiment script"
```

**Note:** RMVPE setup may need debugging. The `rmvpe.py` from different RVC forks has slightly different APIs. If `infer_from_audio` doesn't exist, look for `infer_from_audio_with_pitch` or check the class methods. If setup proves too painful, skip RMVPE and note it in the comparison.

---

### Task 10: Write compare.py

**Files:**
- Create: `experiments/compare.py`

**Step 1: Write the comparison script**

```python
"""Compare all pitch detector results."""
from common import (
    RESULTS_DIR, EXPECTED_TOP_NOTES, EXPECTED_OTHER_NOTES,
    load_results, Frame,
)
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analysis import PitchFrame, build_note_histogram, hz_to_note_name

import numpy as np
import matplotlib.pyplot as plt


DETECTORS = ["crepe", "basicpitch", "swiftf0", "fcpe", "penn", "pesto", "rmvpe"]


def load_all() -> dict[str, dict]:
    """Load all available result files."""
    results = {}
    for name in DETECTORS:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            results[name] = load_results(name)
    return results


def summary_table(results: dict[str, dict]):
    """Print a comparison summary table."""
    print(f"\n{'Detector':<15} {'Time (s)':>8} {'Frames':>8} {'Voiced':>8} {'Top 3 Notes':<25} {'Match?'}")
    print("-" * 80)

    for name, data in results.items():
        frames = [Frame(f["time"], f["freq"], f["confidence"]) for f in data["frames"]]
        pitch_frames = [PitchFrame(f.time, f.freq, f.confidence) for f in frames]
        histogram = build_note_histogram(pitch_frames, confidence_threshold=0.3)
        top3 = sorted(histogram.items(), key=lambda x: -x[1])[:3]
        top3_names = {n for n, _ in top3}
        top3_str = ", ".join(f"{n}({d:.1f}s)" for n, d in top3)
        voiced = sum(1 for f in frames if f.freq > 0)

        # Check if top 3 overlap with expected
        overlap = top3_names & EXPECTED_TOP_NOTES
        match = f"{len(overlap)}/3"

        print(f"{name:<15} {data['elapsed_seconds']:>8.2f} {len(frames):>8} {voiced:>8} {top3_str:<25} {match}")


def consensus_analysis(results: dict[str, dict]):
    """For each time frame, find majority-vote pitch and measure deviations."""
    # Build a common time grid (10ms resolution)
    all_times = set()
    for data in results.values():
        for f in data["frames"]:
            all_times.add(round(f["time"], 2))
    times = sorted(all_times)

    # For each detector, build a time->note lookup
    note_maps = {}
    for name, data in results.items():
        nmap = {}
        for f in data["frames"]:
            if f["freq"] > 0 and f["confidence"] > 0.3:
                t = round(f["time"], 2)
                nmap[t] = hz_to_note_name(f["freq"])
        note_maps[name] = nmap

    # Consensus: majority vote per time step
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
        majority_count = votes[majority_note]
        if majority_count < 2:
            continue  # No consensus

        total_voted += 1
        for name in results:
            note = note_maps[name].get(t)
            if note == majority_note:
                agreement_counts[name] += 1

    print(f"\n{'=' * 50}")
    print(f"  Consensus Agreement ({total_voted} frames with majority)")
    print(f"{'=' * 50}")
    for name, count in sorted(agreement_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_voted * 100) if total_voted > 0 else 0
        print(f"  {name:<15} {count:>6}/{total_voted} ({pct:.1f}%)")


def plot_contours(results: dict[str, dict]):
    """Overlay pitch contours for all detectors."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        times = []
        freqs = []
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
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        frames = [PitchFrame(f["time"], f["freq"], f["confidence"]) for f in data["frames"]]
        histogram = build_note_histogram(frames, confidence_threshold=0.3)
        top = sorted(histogram.items(), key=lambda x: -x[1])[:10]
        if not top:
            continue
        notes, durations = zip(*top)
        colors = ["green" if n in EXPECTED_TOP_NOTES
                  else "orange" if n in EXPECTED_OTHER_NOTES
                  else "steelblue" for n in notes]
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
```

**Step 2: Commit**

```bash
git add experiments/compare.py
git commit -m "feat: add comparison script with summary table and plots"
```

---

### Task 11: Run all experiments and compare

**Step 1: Run all detector scripts**

Run them sequentially (some share GPU memory):

```bash
cd experiments
python separate_vocals.py
python run_crepe.py
python run_basicpitch.py
python run_swiftf0.py
python run_fcpe.py
python run_penn.py
python run_pesto.py
python run_rmvpe.py
```

If any script fails, note the error and move on. The comparison handles missing results gracefully.

**Step 2: Run comparison**

```bash
python compare.py
```

Expected output:
- Summary table with timing, frame counts, top notes, and ground truth match
- Consensus agreement percentages
- `results/contour_comparison.png` — overlay of all pitch contours
- `results/histogram_comparison.png` — side-by-side note histograms

**Step 3: Review results and pick a winner**

Evaluate based on:
1. Do top notes match expected (C4, E4, A3)?
2. Inference speed
3. Pitch contour cleanliness (visual)
4. Consensus agreement percentage

---

## Task Dependencies

```
Task 1 (scaffold) ─┬─> Task 2 (Demucs) ─┬─> Task 3 (CREPE)
                    │                     ├─> Task 5 (SwiftF0)
                    │                     ├─> Task 6 (FCPE)
                    │                     ├─> Task 7 (PENN)
                    │                     └─> Task 8 (PESTO)
                    ├─> Task 4 (BasicPitch)  ──────────┐
                    └─> Task 9 (RMVPE)  ───────────────┤
                                                       v
                                              Task 10 (compare.py)
                                                       v
                                              Task 11 (run all)
```

Tasks 3-9 are independent and can be parallelized. Tasks 3, 5, 6, 7, 8 depend on Task 2 (cached vocals). Tasks 4 and 9 don't need vocals and can run immediately after Task 1.

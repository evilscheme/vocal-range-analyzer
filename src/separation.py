"""Vocal isolation using Demucs."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def isolate_vocals(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "htdemucs",
    device: str | None = None,
) -> tuple[np.ndarray, int]:
    """Separate vocals from an audio file using Demucs.

    Args:
        audio_path: Path to input audio file.
        output_dir: If provided, save isolated vocals WAV here.
        model: Demucs model name.
        device: 'cpu', 'cuda', 'mps', or None (auto-detect).

    Returns:
        Tuple of (vocals_mono_float32, sample_rate).
    """
    try:
        import demucs.api
    except ImportError:
        raise ImportError(
            "Demucs is required for Pipeline A. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-a]'"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    separator = demucs.api.Separator(model=model, device=device)
    _, separated = separator.separate_audio_file(str(audio_path))
    vocals_tensor = separated["vocals"]

    # Convert to mono numpy float32
    vocals_np = vocals_tensor.cpu().numpy()
    if vocals_np.ndim == 2 and vocals_np.shape[0] > 1:
        vocals_np = vocals_np.mean(axis=0)
    elif vocals_np.ndim == 2:
        vocals_np = vocals_np[0]

    sample_rate = separator.samplerate

    if output_dir is not None:
        import soundfile as sf
        out_path = Path(output_dir) / "vocals.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), vocals_np, sample_rate)

    return vocals_np, sample_rate

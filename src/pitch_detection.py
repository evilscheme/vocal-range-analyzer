"""Pitch detection using torchcrepe."""
from __future__ import annotations

import numpy as np

from src.analysis import PitchFrame


def detect_pitch(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 2000.0,
    model: str = "full",
    confidence_threshold: float = 0.21,
    device: str | None = None,
) -> list[PitchFrame]:
    """Detect pitch contour from audio using torchcrepe.

    Args:
        audio: Mono float32 numpy array.
        sample_rate: Sample rate in Hz.
        hop_length: Hop length in samples.
        fmin: Minimum frequency (Hz). Default 65 = C2.
        fmax: Maximum frequency (Hz). Default 2000, well above soprano.
        model: 'full' (accurate) or 'tiny' (fast).
        confidence_threshold: Periodicity threshold for voicing.
        device: 'cpu', 'cuda', 'mps', or None (auto-detect).

    Returns:
        List of PitchFrame objects.
    """
    try:
        import torch
        import torchcrepe
    except ImportError:
        raise ImportError(
            "torchcrepe is required for Pipeline A. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-a]'"
        )

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    pitch, periodicity = torchcrepe.predict(
        audio_tensor,
        sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model,
        return_periodicity=True,
        device=device,
    )

    # Smooth periodicity and apply voicing threshold
    periodicity = torchcrepe.filter.median(periodicity, win_length=3)
    pitch = torchcrepe.threshold.At(confidence_threshold)(pitch, periodicity)

    pitch_np = pitch.squeeze().cpu().numpy()
    periodicity_np = periodicity.squeeze().cpu().numpy()
    hop_seconds = hop_length / sample_rate

    frames: list[PitchFrame] = []
    for i, (f, p) in enumerate(zip(pitch_np, periodicity_np)):
        if np.isnan(f) or f <= 0:
            frames.append(PitchFrame(
                time_seconds=i * hop_seconds,
                frequency_hz=0.0,
                confidence=0.0,
            ))
        else:
            frames.append(PitchFrame(
                time_seconds=i * hop_seconds,
                frequency_hz=float(f),
                confidence=float(p),
            ))

    return frames

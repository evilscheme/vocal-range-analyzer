"""Pitch detection using FCPE (Fast Context-based Pitch Estimation)."""
from __future__ import annotations

import warnings

import numpy as np

from src.analysis import PitchFrame


_FCPE_SR = 16_000
_FCPE_HOP = 160  # 10 ms at 16 kHz


def detect_pitch(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = 65.0,
    fmax: float = 2000.0,
    device: str | None = None,
) -> list[PitchFrame]:
    """Detect pitch contour from audio using FCPE.

    Args:
        audio: Mono float32 numpy array.
        sample_rate: Sample rate in Hz.
        fmin: Minimum frequency (Hz). Default 65 = C2.
        fmax: Maximum frequency (Hz). Default 2000, well above soprano.
        device: 'cpu', 'cuda', 'mps', or None (auto-detect).

    Returns:
        List of PitchFrame objects.
    """
    try:
        import torch
        from torchfcpe import spawn_bundled_infer_model
    except ImportError:
        raise ImportError(
            "torchfcpe is required for Pipeline A. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-a]'"
        )

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # FCPE requires 16 kHz audio
    if sample_rate != _FCPE_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_FCPE_SR)

    model = spawn_bundled_infer_model(device=device)

    # FCPE expects shape (batch, samples, 1)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)
    target_len = (len(audio) // _FCPE_HOP) + 1

    # torchfcpe 0.0.4 calls torch.stft without pre-allocated output,
    # triggering a deprecation warning in PyTorch 2.x about tensor resizing.
    with warnings.catch_warnings(), torch.no_grad():
        warnings.filterwarnings(
            "ignore",
            message="An output with one or more elements was resized",
            category=UserWarning,
        )
        f0 = model.infer(
            audio_tensor,
            sr=_FCPE_SR,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=fmin,
            f0_max=fmax,
            interp_uv=False,
            output_interp_target_length=target_len,
        )

    f0_np = f0.squeeze().cpu().numpy()
    hop_seconds = _FCPE_HOP / _FCPE_SR

    frames: list[PitchFrame] = []
    for i, f in enumerate(f0_np):
        if f > 0:
            frames.append(PitchFrame(
                time_seconds=i * hop_seconds,
                frequency_hz=float(f),
                confidence=1.0,
            ))
        else:
            frames.append(PitchFrame(
                time_seconds=i * hop_seconds,
                frequency_hz=0.0,
                confidence=0.0,
            ))

    return frames

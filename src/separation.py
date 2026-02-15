"""Vocal isolation using Demucs."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_audio(audio_path: Path, target_sr: int) -> np.ndarray:
    """Load audio file as float32 numpy array with shape (channels, samples).

    Uses soundfile directly for supported formats. Falls back to ffmpeg
    for unsupported formats (e.g. M4A/AAC) to avoid librosa's deprecated
    audioread fallback.
    """
    import librosa

    try:
        audio_np, sr = sf.read(str(audio_path), always_2d=True)
    except sf.LibsndfileError:
        # Convert unsupported formats (m4a, aac, wma, etc.) to WAV via ffmpeg
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-ar", str(target_sr),
                 "-ac", "2", "-f", "wav", "-y", tmp_path],
                capture_output=True, check=True,
            )
            audio_np, sr = sf.read(tmp_path, always_2d=True)
        finally:
            if tmp_path:
                os.unlink(tmp_path)

    # audio_np is (samples, channels) from soundfile — transpose to (channels, samples)
    audio_np = audio_np.T.astype(np.float32)

    if sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)

    return audio_np


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
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError:
        raise ImportError(
            "Demucs is required for Pipeline A. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-a]'"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    demucs_model = get_model(model)
    sample_rate = demucs_model.samplerate

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    demucs_model.to(device)

    audio_np = _load_audio(audio_path, sample_rate)
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np])  # mono -> stereo
    wav = torch.tensor(audio_np, dtype=torch.float32)
    wav = wav.unsqueeze(0).to(device)  # (1, channels, samples)

    # Apply model
    with torch.no_grad():
        sources = apply_model(demucs_model, wav, device=device)

    # Extract vocals (index matches model.sources order)
    vocals_idx = demucs_model.sources.index("vocals")
    vocals_tensor = sources[0, vocals_idx]  # (channels, samples)

    # Convert to mono numpy float32
    vocals_np = vocals_tensor.cpu().numpy()
    if vocals_np.ndim == 2 and vocals_np.shape[0] > 1:
        vocals_np = vocals_np.mean(axis=0)
    elif vocals_np.ndim == 2:
        vocals_np = vocals_np[0]

    if output_dir is not None:
        out_path = Path(output_dir) / "vocals.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), vocals_np, sample_rate)

    return vocals_np, sample_rate

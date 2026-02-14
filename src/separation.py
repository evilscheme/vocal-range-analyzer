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
        import torch
        import librosa
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

    # Load audio with librosa (avoids torchaudio/torchcodec issues)
    audio_np, sr = librosa.load(str(audio_path), sr=sample_rate, mono=False)
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
        import soundfile as sf
        out_path = Path(output_dir) / "vocals.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), vocals_np, sample_rate)

    return vocals_np, sample_rate

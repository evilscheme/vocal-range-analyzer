"""Baseline: torchcrepe pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit

import numpy as np


def main():
    enforce_memory_limit(16)
    import torch
    import torchcrepe

    audio = load_vocals(sr=16000)

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
                audio_tensor, 16000,
                hop_length=hop_length, fmin=65.0, fmax=2000.0,
                model="full", return_periodicity=True,
                batch_size=2048, device=device,
            )
        periodicity = torchcrepe.filter.median(periodicity, win_length=3)
        pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)

    pitch_np = pitch.squeeze().cpu().numpy()
    periodicity_np = periodicity.squeeze().cpu().numpy()
    hop_sec = hop_length / 16000

    frames = []
    for i, (f, p) in enumerate(zip(pitch_np, periodicity_np)):
        if np.isnan(f) or f <= 0:
            frames.append(Frame(i * hop_sec, 0.0, 0.0))
        else:
            frames.append(Frame(i * hop_sec, float(f), float(p)))

    save_results("crepe", frames, t["elapsed"], {"model": "full", "hop_length": hop_length})
    print_summary("CREPE (baseline)", frames, t["elapsed"])


if __name__ == "__main__":
    main()

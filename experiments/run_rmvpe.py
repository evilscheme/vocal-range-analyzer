"""RMVPE pitch detection on full mix (no separation needed)."""
from common import load_audio, AUDIO_FILE, save_results, print_summary, Frame, timer, enforce_memory_limit

import sys
from pathlib import Path

# Add rmvpe_model to path so we can import the RMVPE class
sys.path.insert(0, str(Path(__file__).parent / "rmvpe_model"))


def main():
    enforce_memory_limit(16)
    import torch
    from rmvpe import RMVPE

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # RMVPE with is_half=True requires CUDA; use False for CPU/MPS

    model_path = str(Path(__file__).parent / "rmvpe_model" / "rmvpe.pt")
    model = RMVPE(model_path, is_half=False, device=device)

    # RMVPE works on full mix — no separation needed
    audio = load_audio(AUDIO_FILE, sr=16000)

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

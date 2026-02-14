"""PENN (FCNF0++) pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    import torch
    import penn

    audio = load_vocals(sr=16000)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    gpu = None  # CPU by default
    if torch.cuda.is_available():
        gpu = 0

    with timer() as t:
        pitch, periodicity = penn.from_audio(
            audio_tensor, 16000,
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

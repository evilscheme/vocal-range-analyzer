"""PESTO pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    import torch
    import pesto

    audio = load_vocals(sr=16000)
    audio_tensor = torch.from_numpy(audio).float()  # 1D: (num_samples,)

    with timer() as t:
        timesteps, pitch, confidence, _ = pesto.predict(
            audio_tensor, 16000,
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

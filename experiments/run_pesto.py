"""PESTO pitch detection on separated vocals (chunked to limit memory)."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    import torch
    import pesto
    import gc

    audio = load_vocals(sr=16000)

    # Process in 30-second chunks to avoid PESTO's massive memory usage
    chunk_seconds = 30
    chunk_samples = chunk_seconds * 16000
    all_ts, all_pitch, all_conf = [], [], []

    with timer() as t:
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            audio_tensor = torch.from_numpy(chunk).float()
            offset_sec = start / 16000

            timesteps, pitch, confidence, _ = pesto.predict(
                audio_tensor, 16000,
                step_size=10.0,
                convert_to_freq=True,
            )

            # pesto returns timesteps in milliseconds — convert to seconds
            all_ts.extend((timesteps.cpu().numpy() / 1000.0 + offset_sec).tolist())
            all_pitch.extend(pitch.cpu().numpy().tolist())
            all_conf.extend(confidence.cpu().numpy().tolist())

            del audio_tensor, timesteps, pitch, confidence
            gc.collect()

    frames = [
        Frame(ts, p if p > 0 else 0.0, c)
        for ts, p, c in zip(all_ts, all_pitch, all_conf)
    ]

    save_results("pesto", frames, t["elapsed"], {"step_size_ms": 10.0, "chunk_seconds": chunk_seconds})
    print_summary("PESTO", frames, t["elapsed"])


if __name__ == "__main__":
    main()

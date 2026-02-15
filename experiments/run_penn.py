"""PENN (FCNF0++) pitch detection on separated vocals (chunked to limit memory)."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    import torch
    import penn
    import gc

    audio = load_vocals(sr=16000)

    gpu = None
    if torch.cuda.is_available():
        gpu = 0

    # Process in 10-second chunks to stay under 16GB
    # (PENN's model + viterbi decoding is very memory-hungry)
    chunk_seconds = 10
    chunk_samples = chunk_seconds * 16000
    all_pitch, all_period = [], []

    with timer() as t:
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)

            pitch, periodicity = penn.from_audio(
                chunk_tensor, 16000,
                hopsize=0.01,
                fmin=65.0, fmax=2000.0,
                batch_size=512,
                decoder='argmax',  # viterbi uses too much memory
                gpu=gpu,
            )

            all_pitch.extend(pitch.squeeze().cpu().numpy().tolist())
            all_period.extend(periodicity.squeeze().cpu().numpy().tolist())

            del chunk_tensor, pitch, periodicity
            gc.collect()

    frames = [
        Frame(i * 0.01, float(p), float(c))
        for i, (p, c) in enumerate(zip(all_pitch, all_period))
    ]

    save_results("penn", frames, t["elapsed"],
                 {"hopsize": 0.01, "fmin": 65.0, "fmax": 2000.0, "chunk_seconds": chunk_seconds})
    print_summary("PENN (FCNF0++)", frames, t["elapsed"])


if __name__ == "__main__":
    main()

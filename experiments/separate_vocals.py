"""Run Demucs separation once and cache vocals WAV."""
from common import AUDIO_FILE, RESULTS_DIR, enforce_memory_limit, timer

from src.separation import isolate_vocals


def main():
    enforce_memory_limit(16)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Separating vocals from: {AUDIO_FILE}")

    with timer() as t:
        vocals, sr = isolate_vocals(AUDIO_FILE, output_dir=RESULTS_DIR)

    print(f"Done in {t['elapsed']:.1f}s")
    print(f"Vocals saved to: {RESULTS_DIR / 'vocals.wav'}")
    print(f"Shape: {vocals.shape}, SR: {sr}")


if __name__ == "__main__":
    main()

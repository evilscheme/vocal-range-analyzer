"""SwiftF0 pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    from swift_f0 import SwiftF0

    audio = load_vocals(sr=16000)
    detector = SwiftF0(confidence_threshold=0.5, fmin=65.0, fmax=2000.0)

    with timer() as t:
        result = detector.detect_from_array(audio, 16000)

    frames = [
        Frame(float(ts), float(p) if v else 0.0, float(c))
        for ts, p, c, v in zip(
            result.timestamps, result.pitch_hz, result.confidence, result.voicing
        )
    ]

    save_results("swiftf0", frames, t["elapsed"],
                 {"confidence_threshold": 0.5, "fmin": 65.0, "fmax": 2000.0})
    print_summary("SwiftF0", frames, t["elapsed"])


if __name__ == "__main__":
    main()

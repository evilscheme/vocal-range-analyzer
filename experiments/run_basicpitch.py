"""Baseline: BasicPitch pitch detection on full mix."""
from common import AUDIO_FILE, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    with timer() as t:
        _, _, note_events = predict(
            str(AUDIO_FILE),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=50.0,
            minimum_frequency=65.0,
            maximum_frequency=2000.0,
        )

    HOP = 0.01
    frames = []
    for start, end, midi_pitch, amplitude, _bends in note_events:
        freq = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
        t_pos = start
        while t_pos < end:
            frames.append(Frame(t_pos, freq, float(amplitude)))
            t_pos += HOP

    # Deduplicate overlapping notes (keep highest confidence per time slot)
    frames.sort(key=lambda f: (f.time, -f.confidence))
    deduped = []
    seen = set()
    for f in frames:
        key = round(f.time, 3)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    frames = deduped

    save_results("basicpitch", frames, t["elapsed"])
    print_summary("BasicPitch (baseline)", frames, t["elapsed"])


if __name__ == "__main__":
    main()

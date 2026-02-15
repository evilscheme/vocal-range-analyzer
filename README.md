# Vocal Range Analyzer

Analyze the vocal range of a song from an audio file. Outputs the highest/lowest notes, total range, and a duration-weighted note histogram with voice type overlay.

Uses Demucs for vocal isolation and FCPE for pitch detection.

## Example Output

```
==================================================
  Demucs + FCPE
==================================================
  Lowest note:  G3 (MIDI 55)
  Highest note: A4 (MIDI 69)
  Range:        1 octave + 2 semitones

  Top notes by duration:
      E4: 13.86s #####################################################################
      A3: 11.28s ########################################################
      C4:  6.51s ################################
     D#4:  6.05s ##############################
      D4:  4.88s ########################
      G4:  3.71s ##################
     G#3:  3.59s #################
      B3:  2.78s #############
     G#4:  2.14s ##########
      A4:  1.78s ########
```

When `--output` is specified, the output directory will contain:
- `vocals.wav` — isolated vocal track
- `karaoke.wav` — accompaniment (everything minus vocals)
- `range.png` — note histogram with voice type overlays
- `contour.png` — pitch contour plot

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Include dev dependencies (pytest)
uv sync --group dev
```

## Usage

```bash
uv run python analyze.py song.mp3 --output results/

# Adjust confidence threshold (higher = stricter, fewer notes)
uv run python analyze.py song.mp3 --confidence 0.6 --output results/

# Skip chart generation
uv run python analyze.py song.mp3 --no-plot
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | none | Output directory for charts, isolated vocals, and karaoke track |
| `--device` | auto | Device for ML inference: `cpu`, `cuda`, `mps` |
| `--confidence` | `0.5` | Confidence threshold for note detection |
| `--no-plot` | off | Skip chart generation |

## How It Works

1. **Demucs** separates the audio into stems (vocals, drums, bass, other)
2. **FCPE** runs pitch detection on the isolated vocal track
3. `PitchFrame` objects (time, frequency, confidence) feed into analysis code that:

- Converts frequencies to MIDI note numbers and note names
- Builds a duration-weighted histogram of note occurrence
- Groups consecutive same-note frames into discrete note events
- Trims outliers by percentile to find the usable vocal range
- Formats the range as octaves + semitones

## Running Tests

```bash
uv run pytest tests/ -v
```

40 unit tests cover Hz/MIDI conversion, histogram building, note event grouping, vocal range computation, and visualization output.

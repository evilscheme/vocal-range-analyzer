# Vocal Range Analyzer

Analyze the vocal range of a song from an audio file. Outputs the highest/lowest notes, total range, and a duration-weighted note histogram with voice type overlay.

Two analysis pipelines are available:

- **Pipeline A** (Demucs + torchcrepe) — Isolates vocals first with Demucs, then runs pitch detection on the clean vocal track. More accurate, slower.
- **Pipeline B** (BasicPitch) — Runs Spotify's BasicPitch directly on the full mix. Faster, but may pick up instrumental notes.

## Example Output

```
==================================================
  Pipeline A (Demucs + torchcrepe)
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

Charts are saved when `--output` is specified: a note histogram with voice type overlays and a pitch contour plot.

## Installation

Requires Python 3.10+.

```bash
# Core + Pipeline A (recommended)
pip install -e ".[pipeline-a,dev]"

# Core + Pipeline B
pip install -e ".[pipeline-b,dev]"

# Everything
pip install -e ".[all,dev]"
```

**Note:** BasicPitch (Pipeline B) requires `scipy<1.12` due to a deprecated API it depends on.

## Usage

```bash
# Analyze with Pipeline A (Demucs vocal separation + torchcrepe pitch detection)
python analyze.py song.mp3 --pipeline a --output results/

# Analyze with Pipeline B (BasicPitch, faster but less accurate)
python analyze.py song.mp3 --pipeline b --output results/

# Run both pipelines and compare
python analyze.py song.mp3 --pipeline both --output results/

# Adjust confidence threshold (higher = stricter, fewer notes)
python analyze.py song.mp3 --pipeline a --confidence 0.6 --output results/

# Skip chart generation
python analyze.py song.mp3 --pipeline a --no-plot
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `a` | Pipeline to use: `a`, `b`, or `both` |
| `--output`, `-o` | none | Output directory for charts and isolated vocals |
| `--device` | auto | Device for ML inference: `cpu`, `cuda`, `mps` |
| `--confidence` | `0.5` | Confidence threshold for note detection |
| `--no-plot` | off | Skip chart generation |

## How It Works

### Pipeline A

1. **Demucs** separates the audio into stems (vocals, drums, bass, other)
2. **torchcrepe** runs pitch detection on the isolated vocal track
3. Pitch frames are grouped into notes and analyzed

### Pipeline B

1. **BasicPitch** analyzes the full mix directly, detecting note onsets and pitches
2. MIDI note events are converted to pitch frames
3. Same analysis as Pipeline A from that point on

### Analysis

Both pipelines produce a list of `PitchFrame` objects (time, frequency, confidence), which feed into shared analysis code that:

- Converts frequencies to MIDI note numbers and note names
- Builds a duration-weighted histogram of note occurrence
- Groups consecutive same-note frames into discrete note events
- Trims outliers by percentile to find the usable vocal range
- Formats the range as octaves + semitones

## Running Tests

```bash
pytest tests/ -v
```

40 unit tests cover Hz/MIDI conversion, histogram building, note event grouping, vocal range computation, and visualization output.

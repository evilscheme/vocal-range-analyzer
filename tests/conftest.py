"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def sine_440hz_wav(tmp_path_factory) -> Path:
    """Generate a 3-second 440Hz (A4) sine wave WAV file."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path_factory.mktemp("fixtures") / "sine_440hz.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture(scope="session")
def two_tone_wav(tmp_path_factory) -> Path:
    """Generate a 4-second WAV: 2s of C4 (261.63Hz), then 2s of A4 (440Hz)."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    c4 = (0.5 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    a4 = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio = np.concatenate([c4, a4])
    path = tmp_path_factory.mktemp("fixtures") / "two_tone.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture(scope="session")
def silence_wav(tmp_path_factory) -> Path:
    """Generate a 2-second silent WAV file."""
    sr = 44100
    duration = 2.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    path = tmp_path_factory.mktemp("fixtures") / "silence.wav"
    sf.write(str(path), audio, sr)
    return path

"""Immutable audio document loaded from disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioDocument:
    """GUI-independent snapshot of decoded PCM."""

    id: str
    samples: np.ndarray  # float32, C-contiguous, shape (n, 1)
    sample_rate: int
    path: Path

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]


def load_audio(path: Path | str) -> AudioDocument:
    """Decode an entire mono audio file into float32 samples shaped (n, 1).

    Formats supported by libsndfile (via soundfile), including WAV and FLAC.
    """
    path = Path(path)
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)

    if data.shape[1] != 1:
        raise ValueError(
            f"Expected mono audio, got {data.shape[1]} channels in {path}"
        )

    samples = np.ascontiguousarray(data, dtype=np.float32)
    return AudioDocument(
        id=str(uuid4()),
        samples=samples,
        sample_rate=int(sample_rate),
        path=path,
    )

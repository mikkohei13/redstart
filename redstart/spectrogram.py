"""Full-file STFT for spectrogram display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from redstart.audio_document import AudioDocument

# Matches ARCHITECTURE.md defaults.
FFT_SIZE = 2048
HOP = 512
WINDOW = "hann"
DB_FLOOR = 1e-10
VIEWPORT_SECONDS = 60.0
FREQ_MAX_HZ = 20_000.0


@dataclass(frozen=True)
class SpectrogramData:
    """Full-document dB power image."""

    db: np.ndarray  # float32, shape (n_freqs, n_frames), row-major
    freqs_hz: np.ndarray
    times_s: np.ndarray  # frame centres in document seconds
    duration_s: float


def compute_full(
    document: AudioDocument,
    *,
    freq_max_hz: float = FREQ_MAX_HZ,
) -> SpectrogramData:
    """Compute STFT for the entire document (once, at open time)."""
    sr = document.sample_rate
    duration_s = document.duration_seconds
    chunk = document.samples[:, 0]

    freqs, times_s, zxx = signal.stft(
        chunk,
        fs=sr,
        window=WINDOW,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP,
        nfft=FFT_SIZE,
        boundary=None,
        padded=False,
    )

    power = np.abs(zxx) ** 2

    # Crop frequency bins to the display range.
    freq_mask = freqs <= freq_max_hz
    if not np.any(freq_mask):
        freq_mask = np.ones(freqs.shape, dtype=bool)
    freqs = freqs[freq_mask]
    power = power[freq_mask, :]

    db = 10.0 * np.log10(np.maximum(power, DB_FLOOR))
    db = np.ascontiguousarray(db, dtype=np.float32)

    return SpectrogramData(
        db=db,
        freqs_hz=np.asarray(freqs, dtype=np.float64),
        times_s=np.asarray(times_s, dtype=np.float64),
        duration_s=duration_s,
    )


def viewport_start_for_time(t: float, duration: float = VIEWPORT_SECONDS) -> float:
    """Snap a playhead time to the start of its viewport page."""
    if t < 0:
        return 0.0
    return (int(t // duration)) * duration


def scroll_x_range(
    playhead: float,
    width: float,
    *,
    duration_s: float | None = None,
) -> tuple[float, float]:
    """X range with playhead always at the horizontal center.

    May extend before 0 or past the file end so the playhead stays centered.
    """
    _ = duration_s
    width = max(width, 1e-3)
    half = width / 2.0
    return playhead - half, playhead + half


def page_x_range_for_time(
    playhead: float,
    width: float,
    *,
    duration_s: float,
) -> tuple[float, float]:
    """Page containing playhead: playhead at left of a window of `width`."""
    width = max(width, 1e-3)
    if duration_s <= width:
        return 0.0, max(duration_s, width)
    t0 = viewport_start_for_time(playhead, width)
    # Last page may start earlier so the window stays full-width.
    max_t0 = max(0.0, duration_s - width)
    t0 = min(t0, max_t0)
    return t0, t0 + width


def page_x_range_after_jump(
    playhead: float,
    width: float,
    *,
    duration_s: float,
) -> tuple[float, float]:
    """After playhead hits the right edge, show the next page of `width`."""
    return page_x_range_for_time(playhead, width, duration_s=duration_s)

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
# Frames per FFT batch: bounds temporaries without needing a progressive UI.
_BLOCK_FRAMES = 4096


@dataclass(frozen=True)
class SpectrogramData:
    """Full-document dB power image."""

    db: np.ndarray  # float32, shape (n_freqs, n_frames), row-major
    freqs_hz: np.ndarray
    times_s: np.ndarray  # frame centres in document seconds


def compute_full(
    document: AudioDocument,
    *,
    freq_max_hz: float = FREQ_MAX_HZ,
) -> SpectrogramData:
    """Compute STFT for the entire document (once, at open time).

    Fills a preallocated dB image in frame blocks so peak memory stays near the
    output size, instead of materialising a full complex STFT at once.
    Scaling matches ``scipy.signal.stft`` (``scaling='spectrum'``).
    """
    sr = document.sample_rate
    x = document.samples[:, 0]

    n_frames = max(0, (len(x) - FFT_SIZE) // HOP + 1)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sr)
    n_bins = int(np.searchsorted(freqs, freq_max_hz, side="right"))
    if n_bins <= 0:
        n_bins = len(freqs)
    freqs = freqs[:n_bins]

    if n_frames == 0:
        return SpectrogramData(
            db=np.empty((n_bins, 0), dtype=np.float32),
            freqs_hz=np.asarray(freqs, dtype=np.float64),
            times_s=np.empty(0, dtype=np.float64),
        )

    window = signal.get_window(WINDOW, FFT_SIZE).astype(np.float64)
    # Same "spectrum" scaling as scipy.signal.stft.
    scale = 1.0 / window.sum()

    # Strided view over the PCM: no extra full-file copy of the samples.
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::HOP]
    db = np.empty((n_bins, n_frames), dtype=np.float32)

    for start in range(0, n_frames, _BLOCK_FRAMES):
        stop = min(start + _BLOCK_FRAMES, n_frames)
        # Window in float64, then rfft; crop bins before power/log.
        block = frames[start:stop].astype(np.float64, copy=False) * window
        spec = np.fft.rfft(block, n=FFT_SIZE, axis=1)[:, :n_bins]
        spec *= scale
        power = spec.real * spec.real + spec.imag * spec.imag
        np.maximum(power, DB_FLOOR, out=power)
        np.log10(power, out=power)
        power *= 10.0
        db[:, start:stop] = power.T

    times_s = (np.arange(n_frames, dtype=np.float64) * HOP + FFT_SIZE / 2.0) / sr

    return SpectrogramData(
        db=db,
        freqs_hz=np.asarray(freqs, dtype=np.float64),
        times_s=times_s,
    )


def scroll_x_range(playhead: float, width: float) -> tuple[float, float]:
    """X range with playhead always at the horizontal center.

    May extend before 0 or past the file end so the playhead stays centered.
    """
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
    # Snap to page start; last page may start earlier so the window stays full-width.
    t0 = max(0.0, (int(playhead // width)) * width)
    t0 = min(t0, max(0.0, duration_s - width))
    return t0, t0 + width

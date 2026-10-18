"""Offline band-pass isolation of a time × frequency selection."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Minimum selection size so the filter has something sensible to work with.
_MIN_DURATION_S = 0.02
_MIN_BANDWIDTH_HZ = 20.0
_FADE_S = 0.008
_FILTER_ORDER = 4
# Extra samples so filtfilt has room at the edges (order-dependent).
_PAD_S = 0.05


class IsolationError(ValueError):
    """Selection is too small or frequency bounds are invalid."""


def bandpass_slice(
    samples: np.ndarray,
    sample_rate: int,
    t0: float,
    t1: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """Return a float32 mono band-passed slice for ``[t0, t1]`` × ``[f_low, f_high]``.

    ``samples`` is mono float PCM shaped ``(n,)`` or ``(n, 1)``. Edge fades avoid
    clicks when looping. Raises :class:`IsolationError` for invalid selections.
    """
    if sample_rate <= 0:
        raise IsolationError("Invalid sample rate.")

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = mono.shape[0]
    duration_s = n / sample_rate

    t0 = max(0.0, min(float(t0), duration_s))
    t1 = max(0.0, min(float(t1), duration_s))
    if t1 < t0:
        t0, t1 = t1, t0
    if t1 - t0 < _MIN_DURATION_S:
        raise IsolationError("Selection is too short.")

    nyquist = 0.5 * sample_rate
    f_low = max(0.0, float(f_low))
    f_high = max(0.0, float(f_high))
    if f_high < f_low:
        f_low, f_high = f_high, f_low
    # Keep passband inside (0, Nyquist).
    f_low = max(f_low, 1.0)
    f_high = min(f_high, nyquist * 0.99)
    if f_high - f_low < _MIN_BANDWIDTH_HZ:
        raise IsolationError("Frequency band is too narrow.")

    i0 = int(round(t0 * sample_rate))
    i1 = int(round(t1 * sample_rate))
    i0 = max(0, min(i0, n))
    i1 = max(0, min(i1, n))
    if i1 <= i0:
        raise IsolationError("Selection is too short.")

    pad = int(round(_PAD_S * sample_rate))
    pad_start = max(0, i0 - pad)
    pad_end = min(n, i1 + pad)
    chunk = mono[pad_start:pad_end].astype(np.float64, copy=True)

    sos = butter(
        _FILTER_ORDER,
        [f_low, f_high],
        btype="band",
        fs=sample_rate,
        output="sos",
    )
    filtered = sosfiltfilt(sos, chunk)

    # Trim padding back to the requested interval.
    local0 = i0 - pad_start
    local1 = i1 - pad_start
    out = np.ascontiguousarray(filtered[local0:local1], dtype=np.float32)

    fade = min(int(round(_FADE_S * sample_rate)), out.shape[0] // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]

    return out

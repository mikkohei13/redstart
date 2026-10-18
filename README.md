# Redstart

Desktop spectrogram player for long nature-sound WAV recordings: scrollable spectrogram view with play/pause/seek synced to the playhead.

## How it’s built

Python desktop app (PySide6 + pyqtgraph). Audio is decoded into an immutable NumPy document (`soundfile`); spectrograms are computed in background STFT chunks (`SciPy`/`NumPy`); playback runs on a PortAudio callback (`sounddevice`) with DAC-timed audible position. A main-thread controller wires UI intents to those services.

## Run

```bash
uv sync
uv run python -m redstart audio/thorsvik/data/20260418_040000.WAV
```

Omit the path to open a file dialog (defaults to `audio/thorsvik/data`).

## Possible future features

- Read bird detection metadata from CSV file
- Save annotations to CSV/JSON file
- Export audio files
- Export observations
- Cache spectrogram locally

## Note: playhead vs audible sync

The UI playhead can lead the sound by a small amount (often tens of ms).

**Cause.** The audio callback advances a write cursor when samples are *submitted* to PortAudio, but they are heard only after output buffering. We already estimate audible time from each buffer’s `outputBufferDacTime` and `stream.time` (see `redstart/playback.py`). Residual error remains because DAC timestamps are estimates, the UI samples position only ~30 Hz, and each spectrogram column is an STFT window (~43 ms at 48 kHz / 2048), so visual onsets are inherently fuzzy.

**Options if this becomes irritating:**

1. **Calibrated offset** — subtract a small fixed delay (ms) from the reported audible position, applied consistently to playhead, pause snap-back, and seek display. Simple; retune per output device/route (speakers vs Bluetooth).
2. **Do not double-count** — keep DAC-based mapping as the base; a fixed offset is only for residual lag, not a replacement for it, and should not also add full `stream.latency` on top.
3. **Smoother UI** — faster timer or clock extrapolation between snapshots (helps jitter, not true audio lag).
4. **Loopback calibration** — play a click, measure input delay (accurate, heavier UX).

A fixed playhead delay alone is fine for steady playback on one device if it matches the residual; it backfires on device changes, if applied only to drawing (not transport), or if stacked on an already-correct DAC estimate.

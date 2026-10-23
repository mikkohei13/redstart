# Redstart

Desktop spectrogram player for long nature-sound WAV recordings. Open a mono WAV, see a scrollable time–frequency view, and play with a playhead kept in sync with what you hear.

**Main features**

- Spectrogram viewport (default ~60 s wide) with play/pause/seek; click the plot to jump
- Scroll and page view modes (playhead-centered scrolling, or paging when the playhead leaves the window)
- Full-file overview strip: detections per minute, with a draggable playhead
- Companion CSV (BirdNET format) overlays (confidence threshold filter) when a matching CSV sits next to the WAV
- Isolation tool: draw a time × frequency box, band-pass that region, and loop-play the selection
- Display settings: colormap, dB floor/ceiling, contrast (persisted via Qt settings)

## How it’s built

Python 3.11+ desktop app. UI is **PySide6**; the spectrogram plot is **pyqtgraph**. Audio I/O is **soundfile** (decode) and **sounddevice**/PortAudio (playback). DSP uses **NumPy** and **SciPy**.

Layout of the `redstart` package:

| Module | Role |
|--------|------|
| `app` / `__main__` | CLI entry, `QApplication`, optional WAV path |
| `audio_document` | Immutable mono PCM snapshot (`float32`, shape `(n, 1)`) |
| `spectrogram` | Full-file STFT → dB image + time/freq axes |
| `playback` | `PlaybackEngine`: one `OutputStream`, transport, audible position |
| `detections` | CSV parse, lane layout, per-minute counts |
| `isolate` | Offline Butterworth band-pass of a time×freq selection |
| `settings` / `settings_dialog` | Persisted display/view prefs (`QSettings`) |
| `main_window` | Main-thread controller: load, view, overlays, isolation UX |

**Data path.** Opening a file decodes the whole WAV into an `AudioDocument`, then computes one STFT for the first channel (Hann 2048, hop 512, spectrum scaling). Frames are processed in blocks so peak memory stays near the output dB array rather than a full complex STFT. The result is a frozen `SpectrogramData` shown as a row-major `ImageItem`.

**Playback.** `PlaybackEngine` owns audible time. The PortAudio callback only copies PCM into the output buffer (no Qt, no DSP). Snapshots report *audible* position by anchoring each submitted buffer to `outputBufferDacTime` and interpolating with `stream.time`. Isolation replaces the engine buffer with a filtered mono loop until the tool is closed.

**UI loop.** A ~30 Hz `QTimer` samples playback snapshots and updates the playhead / driven X-range. Manual pan disables follow in page mode; clicks seek. Detection overlays and the overview bar chart are rebuilt when a companion CSV is found or the confidence threshold is applied. Qt-free document/DSP types stay separate from the window so workers or alternate UIs could reuse them later.

## Run

```bash
uv sync
uv run python -m redstart audio/thorsvik/data/20260418_040000.WAV
```

Omit the path to open a file dialog (defaults to `audio/thorsvik/data`).

## Possible future features

- Show x-axis always as seconds, now shows sometimes ks
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


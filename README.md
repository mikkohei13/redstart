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

- Show x-axis always as seconds, now shows sometimes ks
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

## Notes

The core problem is BirdNET-style overlapping 3 s windows: the same species repeats every 2.5 s, and different species stack in the same stretch. Lane stacking shows every window, which is precise but noisy for “what’s here?”

Here are options that tend to work well, from lightest change to richer overview:

1. Merge consecutive same-species hits
Collapse overlapping/adjacent windows of the same species into one span (start of first → end of last, keep max confidence).
Parus major at 60–70 becomes one bar + one label instead of four. That alone removes most visual clutter while keeping different species separate.

2. Species color coding
Assign a stable color per scientific name (hash or palette). Same species reads as one visual stream even when lanes differ. Pair with short labels (P. major 0.99) or initials on the spectrogram and full names elsewhere.

3. “Now playing” species strip
A thin panel above/below the spectrogram listing species present in the current viewport (or ±N seconds around the playhead), sorted by max confidence, with count or peak score. Updates as you scroll/play. Spectrogram ticks stay sparse; the strip answers “who’s here?”

4. Species swimlanes (Gantt under the plot)
Fixed row per species for the loaded file (or for species above threshold). Each detection is a horizontal bar on its row. Dense periods become clear as “many bars on few rows,” not a pile of text on the spectrogram. Click a row to filter spectrogram overlays to that species.

5. Filter / focus on one species
Checklist or combobox of species in the file. Default “All”; picking one shows only that species’ ticks. Best for verification; combine with (3) or (4) for discovery.

6. Cluster into “events”
Group detections that overlap in time into an event: e.g. at t=310–318 show one chip
Larus canus · Fulica atra (or “2 spp”) and expand on hover/click. Overview stays calm; detail on demand.

7. Confidence as visual weight, not text
Draw semi-transparent bars (opacity ∝ confidence); only label the top-N per time cluster or peaks. Lower-confidence duplicates stay as faint ticks without names.

8. Whole-file species timeline (minimap)
A second, short plot: time × species (or stacked presence). Gives hour-scale “when does gull vs tit dominate?” without reading every overlay. Click jumps the spectrogram.

Practical combo for Redstart (good overview without a large redesign):

Merge same-species runs (1)
Color by species (2)
Viewport species list (3) with click-to-filter (5)
That keeps the spectrogram as the time/freq view, while the list answers presence at a glance. Swimlanes (4) are the next step if you want a dedicated overview panel.

If you want one of these implemented, switch to Agent mode and say which combo you prefer.
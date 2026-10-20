# Desktop Spectrogram App: Technical Draft

Important! This is only a draft / proposal for how to make the app. It can be deviated from if app features, code quality, or iterative development justifies it.

This is also somewhat outdated compared to current codebase. No tiling is needed for now.

## Scope

Build a Python desktop app for WAV/MP3 recordings up to 3,600 seconds. Show a default 60-second spectrogram viewport with playback-synchronized scrolling. Priorities: fast development on macOS and standalone macOS/Windows distribution.

Stack: PySide6, pyqtgraph, NumPy, SciPy, soundfile/libsndfile, sounddevice/PortAudio. Pin dependencies in a lockfile.

V1: mono/stereo loading, progressive spectrograms, play/pause/seek, analysis settings, colour/dB controls, frequency scaling, manual navigation, and follow-playhead mode. Defer selection playback, processors, ML, comparison layers, and a whole-file overview; preserve their extension boundaries.

## 1. Audio document

- GUI-independent, immutable `AudioDocument`: ID, C-contiguous `float32` samples shaped `(samples, channels)`, source sample rate, and metadata.
- Decode the entire file in a worker using block reads; preallocate when frame counts are reliable. Report progress/errors and support cooperative cancellation. Validate duration against decoded samples.
- Publish only complete documents. Playback becomes available after decoding and does not wait for spectrogram completion.
- Preserve original samples/rate. Reject unsupported channel layouts explicitly. Preflight decoded memory (`samples * channels * 4`) against a configurable budget; avoid duplicate full-file arrays.

## 2. Spectrogram service

- Input: document snapshot and immutable analysis settings. Output: `float32` dB power cache, frame/frequency coordinates, and completed-range metadata.
- Defaults: Hann window, FFT/window size 2,048, hop 512, first-channel analysis. Channel policy is explicit; playback preserves stereo.
- Compute bounded chunks on a global frame grid. Frame centres: `frame_index * hop / sample_rate`. Include window context across chunks; pad only at file endpoints.
- Use consistent power/window normalization, a fixed documented dB reference, and a positive floor before logarithms. Never normalize chunks independently.
- Cache key: document ID, channel policy, FFT/window/hop settings, normalization version. Settings changes create a new generation and cancel the old job.
- Publish completed ranges only. Discard stale-generation results; never reuse storage while an old worker can write to it.
- Estimate cache bytes: approximately `4 * (fft_size / 2 + 1) * frame_count` per analysed channel. Use RAM below a configurable threshold and memory-mapped temporary storage above it. Bound FFT temporaries; release caches after consumers finish.
- Display dB data cannot support inverse reconstruction. Future spectral processors compute a local complex STFT from original PCM.

## 3. Playback engine

- Own one `sounddevice.OutputStream`; copy prepared immutable PCM into its output buffer and zero-fill any remainder.
- Callback restrictions: no decoding/DSP/ML, disk I/O, logging, Qt calls/signals, blocking synchronization, or large allocations.
- Apply transport/source commands through an explicit handoff at callback boundaries. Never block the callback; retain old buffers until switching is acknowledged.
- Own actual playback position. Distinguish submitted samples from audible position using source offsets, `outputBufferDacTime`, and the stream clock. Publish consistent snapshots; reset mappings on seeks/source changes and handle queued output explicitly.
- Define EOF, underflow reporting, device errors, channel mapping, and short discontinuity fades. Validate device sample rates; prepare resampling outside the callback with an explicit source-time mapping.
- Accept prepared playback sources. Original PCM is the V1 source; processed regions later provide the same interface plus their document-time origin.

## 4. View layer

- Create/update widgets and graphics items only on the Qt main thread.
- Supply C-contiguous row-major tiles and explicit levels. Reuse images during scrolling; do not rebuild them on every timer tick.
- A 30–60 Hz timer samples playback snapshots and updates the playhead/view range. It never advances playback time or computes spectra.
- Colour/dB controls affect rendering only. Log-frequency display requires image resampling/custom mapping, not just log ticks; retain the STFT cache. Maintain inverse coordinate mappings for future selections.
- Manual navigation can disable follow mode.

## 5. Controller and concurrency

- Main-thread controller owns document lifecycle, requested transport actions, settings, viewport, and follow mode. Actual playback position comes from the engine.
- Views emit intents; the controller invokes service APIs. Worker notifications use queued Qt signals. Keep document types/DSP independent of Qt.
- Use three execution domains: UI, audio callback, bounded worker pool. Native libraries may create additional threads.
- Signals do not copy NumPy arrays or establish ownership. Workers exclusively own unfinished output; published regions are read-only and never rewritten.
- Use generation IDs, cooperative cancellation, consistent snapshots, and explicit buffer lifetimes. Document replacement/shutdown invalidates jobs, stops playback safely, and defers cleanup until consumers release references.

## Deferred processor contract

A processor receives source audio/rate/layout, optional selection, parameters, context requirements, and cancellation. Selections use half-open source-sample intervals and Hz bounds. Results contain immutable audio, output rate/layout, document-time origin, delay/alignment, and provenance.

Start with band-pass selection playback using surrounding context and boundary fades. Spectral masking alternatively requires complex STFT and valid inverse reconstruction. Frequency isolation has transition bands and temporal tails.

Add a small built-in registry when processors are implemented. ML remains outside core dependencies; use a separate process if needed, with shared-memory/file-backed inputs. Optional ML installation in a standalone app requires a separately packaged runtime/helper or ML-enabled build.

## Delivery and verification

1. Implement decode -> visible tiles (no?) -> transport -> synchronized scrolling. Run directly from the Python environment during development.
2. Package early with PyInstaller on each target OS; include Qt plugins, PortAudio, and libsndfile. Test without Python installed. Plan macOS signing/notarization, Windows installer/signing, and explicit Apple Silicon/Intel support.
3. Verify chunked STFT against whole-signal output, coordinate/timing mappings, stale-result rejection, cancellation, EOF, device failures, and shutdown.
4. Benchmark one-hour input during analysis/playback/seeking/scrolling: peak memory, load/analysis latency, UI responsiveness, and underflows. Verify packaged WAV/MP3 decoding on both platforms.


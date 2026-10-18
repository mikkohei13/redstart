"""Realtime PCM playback via sounddevice."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np
import sounddevice as sd

from redstart.audio_document import AudioDocument


@dataclass(frozen=True)
class PlaybackSnapshot:
    position_seconds: float
    position_samples: int
    playing: bool
    finished: bool


class PlaybackEngine:
    """Owns one OutputStream. Actual audible position lives here, not in the UI.

    The callback advances a write cursor when samples are *submitted* to PortAudio.
    Audible position lags that by the output buffer latency. Snapshots report the
    audible position by anchoring each submitted buffer to ``outputBufferDacTime``
    and interpolating with ``stream.time``.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._samples: np.ndarray | None = None  # float32 (n,) mono
        self._sample_rate = 0
        self._cursor = 0  # next sample to submit
        self._playing = False
        self._finished = False
        self._stream: sd.OutputStream | None = None
        # Mapping: source sample at buffer start <-> DAC time of that sample.
        self._dac_time_anchor: float | None = None
        self._source_sample_anchor = 0

    def load(self, document: AudioDocument) -> None:
        self.stop()
        with self._lock:
            self._samples = document.samples[:, 0]
            self._sample_rate = document.sample_rate
            self._cursor = 0
            self._playing = False
            self._finished = False
            self._clear_timing_anchor()

    def play(self) -> None:
        with self._lock:
            if self._samples is None:
                return
            if self._finished:
                self._cursor = 0
                self._finished = False
            self._playing = True
            self._clear_timing_anchor()
            sample_rate = self._sample_rate

        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency="low",
            )
            self._stream.start()

    def pause(self) -> None:
        # Snap the write cursor back to what was actually heard so resume
        # does not skip the still-buffered latency window.
        audible = self._audible_samples()
        with self._lock:
            self._playing = False
            if audible is not None:
                self._cursor = audible
            self._clear_timing_anchor()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            self._playing = False
            self._cursor = 0
            self._finished = False
            self._clear_timing_anchor()

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self._samples is None or self._sample_rate <= 0:
                return
            sample = int(round(seconds * self._sample_rate))
            sample = max(0, min(sample, len(self._samples)))
            self._cursor = sample
            self._finished = sample >= len(self._samples)
            self._clear_timing_anchor()

    def snapshot(self) -> PlaybackSnapshot:
        audible = self._audible_samples()
        with self._lock:
            if self._samples is None or self._sample_rate <= 0:
                return PlaybackSnapshot(0.0, 0, False, False)
            pos = self._cursor if audible is None else audible
            return PlaybackSnapshot(
                position_seconds=pos / self._sample_rate,
                position_samples=pos,
                playing=self._playing,
                finished=self._finished,
            )

    def _clear_timing_anchor(self) -> None:
        self._dac_time_anchor = None
        self._source_sample_anchor = 0

    def _audible_samples(self) -> int | None:
        """Estimate the source sample currently at the DAC, or None if unknown."""
        stream = self._stream
        with self._lock:
            if (
                not self._playing
                or stream is None
                or self._dac_time_anchor is None
                or self._sample_rate <= 0
                or self._samples is None
            ):
                return None
            dac_anchor = self._dac_time_anchor
            source_anchor = self._source_sample_anchor
            write_cursor = self._cursor
            n = len(self._samples)
            sr = self._sample_rate

        # stream.time shares the PortAudio clock with outputBufferDacTime.
        elapsed = stream.time - dac_anchor
        pos = source_anchor + int(elapsed * sr)
        # Cannot have heard samples that have not been submitted yet.
        return max(0, min(pos, write_cursor, n))

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        # Callback restrictions: no Qt, no I/O, no blocking beyond the lock.
        if status:
            pass

        with self._lock:
            if (
                not self._playing
                or self._samples is None
                or self._finished
            ):
                outdata.fill(0)
                return

            remaining = len(self._samples) - self._cursor
            if remaining <= 0:
                outdata.fill(0)
                self._playing = False
                self._finished = True
                self._clear_timing_anchor()
                return

            # Anchor this buffer: first written sample becomes audible at dac time.
            self._dac_time_anchor = float(time.outputBufferDacTime)
            self._source_sample_anchor = self._cursor

            n = min(frames, remaining)
            outdata[:n, 0] = self._samples[self._cursor : self._cursor + n]
            if n < frames:
                outdata[n:].fill(0)
                self._playing = False
                self._finished = True
            self._cursor += n

"""
Local microphone capture.

Captures short audio frames from the default input device. Nothing here
ever writes audio to a network socket — capture stays entirely local, and
frames are handed off in-memory to the ASR engine.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Hz — matches typical ASR model input
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


@dataclass
class AudioFrame:
    samples: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int = SAMPLE_RATE


class MicrophoneStream:
    """Context-managed microphone stream that yields AudioFrame objects."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_samples: int = FRAME_SAMPLES):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            # Non-fatal audio driver warnings land here — log, don't crash.
            print(f"[audio_capture] status: {status}")
        self._queue.put(indata.copy())

    def __enter__(self) -> "MicrophoneStream":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def frames(self):
        """Generator yielding AudioFrame objects as they arrive."""
        while True:
            block = self._queue.get()
            yield AudioFrame(samples=block[:, 0], sample_rate=self.sample_rate)

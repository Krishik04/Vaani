"""
On-device automatic speech recognition.

Intended model: Whisper (tiny/base), INT8-quantized, downloaded from
Qualcomm AI Hub (https://aihub.qualcomm.com/) as an ONNX build optimized
for the Hexagon NPU, and run through ONNX Runtime with the QNN Execution
Provider.

This module currently ships as a stub with the real interface in place so
the rest of the pipeline (disfluency detection, scoring, UI) can be built
and tested end-to-end before the model is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - onnxruntime is a required dependency
    ort = None


@dataclass
class TranscriptSegment:
    text: str
    start_s: float
    end_s: float
    confidence: float


class ASREngine:
    """Wraps an ONNX ASR model running on the Qualcomm QNN Execution Provider."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.session: "ort.InferenceSession | None" = None
        if self.model_path and self.model_path.exists():
            self._load()

    def _load(self) -> None:
        # TODO: point this at a Whisper ONNX build from Qualcomm AI Hub.
        # Providers are tried in order — QNN first (NPU), falling back to
        # CPU on non-Snapdragon dev machines.
        providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        """Transcribe a chunk of audio. Returns an empty list until a real
        model is loaded — replace this method's body once `self.session`
        is wired up to run actual inference.
        """
        if self.session is None:
            # No model loaded yet — safe no-op so the rest of the app can run.
            return []

        # TODO: preprocess `samples` into the model's expected input
        # (e.g. log-mel spectrogram), run `self.session.run(...)`, and
        # decode the output tokens into TranscriptSegment objects.
        raise NotImplementedError("Wire up real Whisper ONNX inference here.")

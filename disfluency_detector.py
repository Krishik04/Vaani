"""
Disfluency detection: flags repetitions, blocks, and prolongations.

Intended approach: a lightweight 1D-CNN or small transformer classifier
over wav2vec2-style acoustic features, trained on open stuttering datasets
(SEP-28k, FluencyBank), exported to ONNX, and quantized (INT8) for
on-device NPU inference — same QNN Execution Provider pattern as
`asr_engine.py`.

Ships as a stub so the pipeline and UI can be developed and tested before
a trained model is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class DisfluencyType(str, Enum):
    REPETITION = "repetition"
    BLOCK = "block"
    PROLONGATION = "prolongation"


@dataclass
class DisfluencyEvent:
    kind: DisfluencyType
    start_s: float
    end_s: float
    confidence: float


class DisfluencyDetector:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.session = None
        if self.model_path and self.model_path.exists():
            self._load()

    def _load(self) -> None:
        import onnxruntime as ort  # local import keeps this optional at dev-time

        providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[DisfluencyEvent]:
        """Return disfluency events found in this audio chunk.

        Returns an empty list until a real model is loaded. Replace the
        body with: extract wav2vec2-style features -> run self.session ->
        threshold/decode into DisfluencyEvent objects.
        """
        if self.session is None:
            return []
        raise NotImplementedError("Wire up the trained disfluency classifier here.")

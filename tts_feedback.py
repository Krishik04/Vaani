"""
On-device text-to-speech for narrating guided exercises and pacing cues.

Intended model: a compact neural TTS model (VITS-class or similar),
run fully on-device. Ships as a stub — swap `speak()`'s body for real
on-device TTS inference once a model is chosen.
"""
from __future__ import annotations

from pathlib import Path


class TTSFeedback:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None

    def speak(self, text: str) -> None:
        """Narrate `text` aloud.

        Stub implementation prints to the console so the rest of the app
        can be exercised without a model in place. Replace with real
        on-device TTS synthesis + playback.
        """
        if self.model_path is None:
            print(f"[tts_feedback] (stub) would narrate: {text!r}")
            return
        raise NotImplementedError("Wire up real on-device TTS synthesis here.")

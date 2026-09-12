"""
Vaani — app entry point.

This is currently a minimal, dependency-light skeleton that wires the full
pipeline together end-to-end (audio -> ASR -> disfluency detection ->
scoring -> storage -> narration) and runs it in a simple command-line loop.
It works today even with no trained models loaded, because every AI module
degrades gracefully to a no-op stub until a real model is wired in — see
the TODOs in asr_engine.py, disfluency_detector.py, and tts_feedback.py.

A PySide6 GUI (live waveform, fluency badge, progress dashboard — matching
the concept UI in the pitch deck) is the natural next step once the
pipeline is verified end-to-end. Swap this file's `main()` for a
QApplication that reuses the same modules below.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from asr_engine import ASREngine
from disfluency_detector import DisfluencyDetector
from fluency_scorer import score_session
from storage import SessionStore
from tts_feedback import TTSFeedback

PRACTICE_SCRIPTS_DIR = Path(__file__).parent.parent / "data" / "practice_scripts"


def load_practice_passage(name: str = "sample_passage_1.txt") -> str:
    path = PRACTICE_SCRIPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def run_practice_session(exercise_name: str = "Paced Reading — Passage 1") -> None:
    print(f"\n=== Vaani — {exercise_name} ===\n")

    passage = load_practice_passage()
    print("Practice passage:\n")
    print(f"  {passage}\n")

    tts = TTSFeedback()
    tts.speak(f"Let's begin. Read at your own pace: {passage}")

    asr = ASREngine()          # TODO: pass model_path once a Whisper ONNX build is in place
    detector = DisfluencyDetector()  # TODO: pass model_path once a trained classifier is in place

    print("[main_app] Listening... (stub: no microphone loop wired in this skeleton)")
    session_start = time.time()

    # In the full app, audio frames stream from MicrophoneStream and are fed
    # to `asr.transcribe(...)` and `detector.detect(...)` continuously. This
    # skeleton simulates that with an empty result set so the rest of the
    # pipeline (scoring, storage) can be exercised without live audio.
    transcript_segments = []  # would come from asr.transcribe(...)
    disfluency_events = []    # would come from detector.detect(...)
    words_spoken = sum(len(seg.text.split()) for seg in transcript_segments) or len(passage.split())

    duration_s = time.time() - session_start

    result = score_session(
        session_duration_s=max(duration_s, 1.0),
        words_spoken=words_spoken,
        events=disfluency_events,
    )

    print(f"\nSession complete — Fluency Score: {result.fluency_score}  "
          f"({result.disfluency_count} disfluencies detected, pacing: {result.pacing_label})\n")

    store = SessionStore()
    store.save_session(
        exercise_name=exercise_name,
        duration_s=duration_s,
        fluency_score=result.fluency_score,
        disfluency_count=result.disfluency_count,
        pacing_label=result.pacing_label,
    )
    print("Recent sessions:")
    for rec in store.recent_sessions(limit=5):
        print(f"  {rec.created_at}  {rec.exercise_name:<32}  score={rec.fluency_score}")
    store.close()


def main() -> None:
    run_practice_session()


if __name__ == "__main__":
    main()

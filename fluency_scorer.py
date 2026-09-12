"""
Combines disfluency events and speech pacing into a per-session fluency score.

This module has no heavy model dependency — it's plain scoring logic that
consumes the outputs of `asr_engine` and `disfluency_detector`. It's fully
functional as written (no TODOs), so it's a good place to start building
and testing the pipeline end-to-end with mocked ASR/disfluency output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from disfluency_detector import DisfluencyEvent


@dataclass
class SessionScore:
    fluency_score: int  # 0-100
    disfluency_count: int
    pacing_label: str
    events: list[DisfluencyEvent] = field(default_factory=list)


def score_session(
    session_duration_s: float,
    words_spoken: int,
    events: list[DisfluencyEvent],
) -> SessionScore:
    """A simple, transparent scoring heuristic.

    fluency_score starts at 100 and is reduced per disfluency event,
    scaled by how many disfluencies occurred relative to words spoken.
    This is intentionally simple and easy to explain to a user — swap in
    a more sophisticated model later if desired.
    """
    if words_spoken <= 0:
        return SessionScore(fluency_score=0, disfluency_count=0, pacing_label="No speech detected")

    disfluency_rate = len(events) / max(words_spoken, 1)
    penalty = min(80, disfluency_rate * 100 * 1.5)
    fluency_score = round(max(0, 100 - penalty))

    words_per_minute = words_spoken / max(session_duration_s / 60, 1e-6)
    if words_per_minute < 90:
        pacing_label = "Slow & Deliberate"
    elif words_per_minute <= 160:
        pacing_label = "Steady"
    else:
        pacing_label = "Fast"

    return SessionScore(
        fluency_score=fluency_score,
        disfluency_count=len(events),
        pacing_label=pacing_label,
        events=events,
    )

# Architecture

## Pipeline

1. **Audio Capture** (`src/audio_capture.py`) — captures microphone input
   locally in short frames, using voice-activity detection to segment speech.
2. **On-Device ASR** (`src/asr_engine.py`) — transcribes speech using a
   quantized Whisper (tiny/base) model, run through ONNX Runtime with the
   Qualcomm QNN Execution Provider so inference is offloaded to the Hexagon
   NPU.
3. **Disfluency Detection** (`src/disfluency_detector.py`) — classifies
   short audio/text windows for repetitions, blocks, and prolongations.
   Intended approach: a lightweight 1D-CNN or small transformer over
   wav2vec2-style acoustic features, trained on open datasets such as
   SEP-28k and FluencyBank, exported to ONNX and quantized for the NPU.
4. **Fluency Scoring** (`src/fluency_scorer.py`) — combines disfluency
   events and speech pacing (via forced alignment of transcript to a target
   script) into a per-session clarity/fluency score and timeline.
5. **Guided Exercise Narration** (`src/tts_feedback.py`) — a compact
   on-device neural TTS model narrates paced-reading passages and breathing
   cues.
6. **Local Storage** (`src/storage.py`) — session history and progress
   scores are stored in a local SQLite database. Nothing is uploaded.
7. **UI** (`src/main_app.py`) — wires the above together into a live
   session view and a progress dashboard.

## Why the Hexagon NPU

| Requirement | Why it matters here |
|---|---|
| Low latency | Feedback needs to feel close to instant to work as a coaching cue |
| Power efficiency | A full practice session shouldn't meaningfully drain the battery |
| Privacy | Therapy-context audio should never need to leave the device |
| Offline availability | Must work without reliable internet, for reach across India |

## Models & sources

| Component | Model / approach | Source |
|---|---|---|
| Speech Recognition (ASR) | Whisper (tiny/base), INT8-quantized | Qualcomm AI Hub–optimized ONNX build |
| Disfluency Detection | Lightweight 1D-CNN / transformer on wav2vec2 features | Trained on open datasets (SEP-28k, FluencyBank) |
| Pacing & Clarity Scoring | CTC-based forced alignment of speech to script | Open-source phoneme aligner |
| Guided Exercise Narration | Compact neural TTS | Open-source TTS (VITS-class) |
| Inference Runtime | ONNX Runtime + Qualcomm QNN Execution Provider | Routes each model to Hexagon NPU / GPU / CPU |
| App Shell & Storage | Python + PySide6, local SQLite | — |

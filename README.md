# Vaani — On-Device AI Speech Fluency Coach

Vaani is a desktop app for Snapdragon-powered PCs that acts as a private,
always-available speech-fluency coach. It listens to the user's speech,
detects disfluencies (repetitions, blocks, prolongations) in real time,
scores fluency and pacing, and guides paced-reading practice — all running
entirely on-device via the Qualcomm Hexagon NPU, with no audio ever leaving
the machine.

Built for the **Snapdragon® AI Lab Build & Present Challenge**.

## Status

This repository is a **starter scaffold**: the app architecture, module
boundaries, and UI flow are in place, with clear interfaces for each AI
component. The model-loading calls are stubbed out with `TODO`s pointing to
where a Qualcomm AI Hub model (or an equivalent open-source model) should be
plugged in. Treat this as the skeleton to build on, not a finished product.

## Architecture

```
Microphone  ─▶  Audio Capture  ─▶  On-Device ASR  ─▶  Disfluency + Fluency   ─▶  Feedback UI
                (audio_capture.py)  (asr_engine.py)     Analysis                  (main_app.py)
                                                        (disfluency_detector.py,
                                                         fluency_scorer.py)
```

All inference is intended to run through **ONNX Runtime with the Qualcomm
QNN Execution Provider**, routing each model to the Hexagon NPU (falling
back to GPU/CPU where unavailable). See `docs/ARCHITECTURE.md` for the full
breakdown of components, models, and sources.

## Project layout

```
vaani/
├── README.md
├── requirements.txt
├── src/
│   ├── main_app.py              # PySide6 app entry point / UI wiring
│   ├── audio_capture.py         # Microphone capture
│   ├── asr_engine.py            # On-device ASR (Whisper via QNN EP) — TODO: plug in model
│   ├── disfluency_detector.py   # Disfluency classifier — TODO: plug in model
│   ├── fluency_scorer.py        # Pacing / clarity scoring
│   ├── tts_feedback.py          # On-device guided-exercise narration — TODO: plug in model
│   └── storage.py               # Local SQLite session history
├── data/
│   └── practice_scripts/        # Sample paced-reading passages
└── docs/
    └── ARCHITECTURE.md
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/main_app.py
```

## Next steps to make this a working build

1. **Download an ASR model from [Qualcomm AI Hub](https://aihub.qualcomm.com/)**
   (e.g. a Whisper tiny/base ONNX build) and point `asr_engine.py` at it.
2. **Train or source a disfluency classifier** — the SEP-28k and FluencyBank
   datasets are commonly used starting points — export it to ONNX, and quantize
   it (INT8) for NPU deployment.
3. **Wire up the QNN Execution Provider** in `onnxruntime.InferenceSession`
   calls (see comments in `asr_engine.py` and `disfluency_detector.py`).
4. **Add a real on-device TTS model** in `tts_feedback.py` for guided-exercise
   narration.
5. Build out the PySide6 UI in `main_app.py` (the current version is a
   minimal CLI-style skeleton so the pipeline can be tested end-to-end first).

## License

MIT — see `LICENSE`.

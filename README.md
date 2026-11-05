# shot-predict-ai

Can a tennis player's pre-contact body motion reveal where their shot will land?

`shot-predict-ai` is a local, offline toolkit for exploring that question. It
covers the whole loop: labeling match footage, training a small temporal model,
and rendering prediction videos.

## What it does

- **Labeling app:** a Streamlit app for marking racket contact and the first
  bounce on match video, with court calibration so each landing click maps to
  one of six receiving-court zones (ad/middle/deuce x deep/short).
- **Semi-automated review queue:** an offline pipeline (court registration,
  YOLO player detection, color/motion ball tracking) proposes shots so you
  confirm or correct them instead of scrubbing through footage.
- **Model training:** a ResNet18 + GRU model that sees only frames *before*
  contact, with match- and clip-grouped splits to prevent leakage.
- **Explainability and export:** Grad-CAM heatmaps, and prediction videos that
  overlay the forecast zone on the match.

## Goals

- Build a reliable, leakage-safe pipeline for collecting and validating shot labels.
- Measure honestly how much shot direction can be predicted from pre-contact
  pose and court position, against simple baselines.
- Get to a system that can forecast placement from live or recorded broadcast video.

## Status: V1

This is an early research prototype. The tooling works end to end, but the
models are **not accurate yet**. On a small held-out set (76 shots) the best CNN
reached about 41% top-1 over six zones, against about 29% for always guessing
the most common zone. Most of that edge comes from recognizing serves. On rally
shots it is roughly at baseline. The results are not calibrated, not
real-time, and not evidence of generalization to other courts or players.

Footage, labels, and trained weights are not included in this repository.

## Quick start

Requires Python 3.12+.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,players]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
.\.venv\Scripts\python.exe -m pytest -q
```

See the [usage guide](docs/USAGE.md) for the full labeling, training, and export
workflow, plus experiment notes and limitations.

## Layout

- `app.py`, `shot_predict/labeler*`: labeling app and frontend
- `shot_predict/autopilot/`: offline shot-proposal and review pipeline
- `shot_predict/{dataset,model,train,predict,explain,players}.py`: training, export, and analysis
- `tests/`: Python and frontend tests

## Responsible use

Only use footage you have permission to use. The app is built for a single local
annotator, so don't expose it publicly.

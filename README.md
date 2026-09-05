---
title: StockCurve
emoji: 📈
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Shazam for Stocks 📈

Sketch-based stock shape retrieval for Indian equity markets (NSE/BSE).

Draw a curve. Get matching stocks.

## How it works

1. User draws a shape on a canvas
2. Shape is normalized and sent to a 3-stage pipeline
3. Each stage narrows the candidate set and streams progress live
4. Top 5 matches returned with overlay charts

## Pipeline

```
All stocks (Nifty 500)
    ↓ Stage 1: Hard filters (cap size, std thresholds)
    ↓ Stage 2: Smoothed Euclidean distance
    ↓ Stage 3a: Fourier (FFT magnitude) matching
    ↓ Convergence: sharpness-adaptive blend
Top 5 results
```

## Setup

```bash
pip install -r requirements.txt  # used python 3.10.10
python prepare_data.py      # run once — downloads data, builds windows.npy
python main.py              # start server
# open http://localhost:8000
```

## For local sharing (ngrok)

```bash
ngrok http 8000
# share the ngrok URL
```

## Deployment (Hugging Face Spaces — free)

The full pipeline (Euclidean + Fourier + Siamese NN + learned fusion) runs on a
free Hugging Face Space: **CPU basic (2 vCPU / 16 GB RAM)**, zero cost. Runtime
data (`windows.npy`, `siamese.pt`, `fusion.pt`) is fetched at startup from the
GitHub release `data-v1` — it stays out of git (see `data/.gitignore`).

One-time setup:
1. Create a read/write token at https://huggingface.co/settings/tokens
2. GitHub repo → Settings → Secrets and variables → Actions → **Secret** `HF_TOKEN` = that token
3. Same page → **Variable** `HF_SPACE_ID` = `your-hf-username/stockcurve`
4. Push to `main` — the `Sync to Hugging Face Space` workflow creates and updates the Space.

Notes:
- The Space sleeps after ~48 h of inactivity (free tier); the next visit re-fetches
  the data files, so the first cold start takes a minute or two.
- The frontend is served by the same FastAPI app, so no CORS setup is needed.
- Same `Dockerfile` + `start.sh` work on any Docker host (Render/Railway) — just
  set the `PORT` env var.

## Project structure

See `AGENTS.md` for full architecture, data schema, and agent rules.

## Research

Every search logs sharpness, alpha blend weight, and per-stage results to stdout.
This data feeds the research paper comparing Fourier vs Euclidean vs (future) ML matching.

## Future work

- Stage 3b: Neural network matcher (see `matchers/nn.py` stub)
- Ablation study: Fourier vs Euclidean vs blend
- Sharpness-adaptive blending evaluation
- User study for match quality ground truth

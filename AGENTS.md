# AGENTS.md — Shazam for Stocks (NSE/BSE Shape Search Engine)

## Project Identity

This is a **sketch-based stock shape retrieval system** for Indian equity markets.
The user draws a curve on a canvas. The system finds stocks whose historical price
curves match that shape. Think Shazam, but for stock chart patterns.

Built by a 2nd-year BS-MS Physics + Data Science student. The codebase must stay
clean, modular, and extensible — future work will add ML/NN matchers and a research
paper comparing approaches.

---



## Agent Rules (Read Before Every Task)

1. **Never break the plugin interface.** Every matcher inherits `BaseMatcher`.
   Adding a new matcher = new file in `matchers/`, register in `MATCHERS` list.
   Never touch the pipeline to add a matcher.

2. **Never couple stages.** Stage 1, 2, 3 are independent. Stage 1 output feeds
   Stage 2, Stage 2 output feeds Stage 3. No stage reaches back.

3. **Stream everything.** Every stage completion must fire an SSE event to the
   frontend. The user must see live progress. Never batch and return at the end.

4. **Python only for computation.** All math — normalization, FFT, distance,
   blending — lives in Python. The frontend does zero computation except
   resampling the canvas points to 50 values before sending.

5. **Keep the frontend dumb.** `index.html` + `style.css` only. No JS frameworks,
   no build steps. JS is only for: canvas drawing, fetch/SSE handling, rendering
   result charts. Nothing else.

6. **One `.npy` file is the database.** `data/windows.npy` holds all precomputed
   windows. `data/meta.json` holds symbol metadata. Nothing else. No SQL, no Redis,
   no external DB.

7. **Free and zero-cost deployment only.** Render free tier for backend. ngrok for
   local dev. No paid services, no paid APIs.

8. **Every function has a docstring.** Single line is fine. No undocumented
   functions.

9. **Stage 3b (ML matcher) is implemented.** `matchers/nn.py` + `matchers/siamese.py`
   provide the Siamese embedding matcher; retrain with `python train.py`. Adding
   new matchers must go through the `BaseMatcher` interface and the `MATCHERS`
   registry — never edit the pipeline to add one.

10. **Do not over-engineer.** If something can be done in 10 lines, do it in 10
    lines. This is a research prototype, not production software.

---
UI rules:
- Use Pico CSS via CDN for base styling. No other CSS framework.
- Write semantic HTML — use <main>, <section>, <article>, <header> properly.
- Custom CSS in style.css only for: canvas styling, result cards, 
  progress indicators, mini charts. Nothing else.
- Dark theme: add data-theme="dark" to <html> tag.
- Mobile is not a priority. Optimize for desktop 1080p+.
- Do not use emojis, if wanted, ask the user for specific icons, else make your own if not    provided, I'll specify when needed

## Project Structure

```
shazam-stocks/
│
├── AGENTS.md                  ← you are here
├── README.md
├── requirements.txt
│
├── data/
│   ├── windows.npy            ← precomputed normalized windows (run prepare_data.py once)
│   ├── meta.json              ← symbol metadata {symbol, name, sector, cap_category}
│   ├── pairs.npy              ← Siamese training pairs (run generate_pairs.py)
│   ├── siamese.pt             ← trained SiameseNet weights (run train.py)
│   ├── fusion.pt              ← trained fusion ranker weights (run train_fusion.py)
│   ├── fusion_config.json     ← fusion feature spec + ablation metrics
│   └── .gitignore             ← ignore generated artifacts (too large for git)
│
├── prepare_data.py            ← run once to build windows.npy and meta.json
├── generate_pairs.py          ← build Siamese training pairs (pairs.npy)
├── train.py                   ← train the Siamese matcher (siamese.pt)
├── evaluate.py                ← matcher evaluation harness
├── train_fusion.py            ← train the learned fusion layer (fusion.pt)
│
├── fusion.py                  ← learned fusion: MLP ranker + features + queries
│
├── matchers/
│   ├── __init__.py            ← MATCHERS registry
│   ├── base.py                ← BaseMatcher abstract class
│   ├── euclidean.py           ← Stage 2: smoothed Euclidean
│   ├── fourier.py             ← Stage 3a: FFT magnitude matching
│   ├── nn.py                  ← Stage 3b: Siamese embedding matcher
│   └── siamese.py             ← SiameseNet architecture (Conv1d encoder)
│
├── pipeline.py                ← Stage 1 filters + convergence layer + fusion/blend
├── main.py                    ← FastAPI app, SSE endpoint, static file serving
│
└── static/
    ├── index.html
    └── style.css
```

---

## Data Schema

### `windows.npy`
Numpy structured array or dict saved with `np.save(..., allow_pickle=True)`.
Each entry:
```python
{
    "symbol":    str,          # e.g. "RELIANCE.NS"
    "date_end":  str,          # e.g. "2024-03-15"
    "cap":       str,          # "large" | "mid" | "small"
    "sector":    str,          # e.g. "Energy"
    "raw":       np.ndarray,   # shape (W,) — original prices, NOT normalized
    "norm":      np.ndarray,   # shape (50,) — resampled + z-normalized
    "smooth":    np.ndarray,   # shape (50,) — norm + gaussian blur sigma=2
    "std":       float,        # std of raw window (for Stage 1 filter)
}
```

### `meta.json`
```json
[
  {
    "symbol": "RELIANCE.NS",
    "name": "Reliance Industries",
    "sector": "Energy",
    "cap": "large"
  }
]
```

### SSE Event Schema
Every stage fires an SSE event. Frontend listens and updates UI per event.
```json
{
  "stage": 1,
  "label": "Hard filters",
  "status": "done",
  "remaining": 241,
  "total": 500,
  "results": []
}
```
Final event (stage 4) includes full top-5 results:
```json
{
  "stage": 4,
  "label": "Final results",
  "status": "done",
  "remaining": 5,
  "total": 500,
  "results": [
    {
      "symbol": "RELIANCE.NS",
      "name": "Reliance Industries",
      "date_end": "2024-03-15",
      "score": 0.87,
      "curve": [0.1, 0.3, ...],
      "matcher_scores": {
        "Euclidean": 0.82,
        "Fourier": 0.91
      }
    }
  ]
}
```

---

## The Algorithm (Pipeline)

### Stage 1 — Hard Filters (pipeline.py)
```
Input: all windows (500 stocks × ~10 windows each = ~5000 total)
Filter by: cap checkboxes from user (large/mid/small bools)
Filter by: std_thresh_low < window.std < std_thresh_high
           (kick flat windows and pure-noise windows)
           defaults: std_thresh_low=0.01, std_thresh_high=3.0
Output: filtered candidate list
Stream: SSE stage=1 event
```

### Stage 2 — Smoothed Euclidean (matchers/euclidean.py)
```
Input: filtered candidates, smoothed sketch
Process: euclidean distance between sketch.smooth and window.smooth
Keep: top 20% of candidates by distance
Output: reduced candidate list with euclidean scores
Stream: SSE stage=2 event
```

### Stage 3a — Fourier Matcher (matchers/fourier.py)
```
Input: reduced candidates, normalized sketch
Process:
  fft_sketch = |FFT(sketch.norm)|[:8]
  fft_window = |FFT(window.norm)|[:8]
  distance = euclidean(fft_sketch, fft_window)
Output: same candidates with fourier scores added
Stream: SSE stage=3 event
```

### Convergence Layer (pipeline.py)
```
Input: candidates with both euclidean_score and fourier_score
Compute sharpness: np.mean(np.abs(np.diff(sketch, n=2)))
alpha = sigmoid(sharpness * 5)   # 0=smooth→Fourier, 1=sharp→Euclidean
final_score = alpha * euclidean_score + (1 - alpha) * fourier_score
Sort by final_score, return top 5
Stream: SSE stage=4 event with full results
```

### Learned Fusion (fusion.py + train_fusion.py) — optional upgrade
When `data/fusion.pt` exists, the convergence layer uses a learned ranker
instead of the heuristic alpha blend (unless NN-matcher availability differs
from the training config, in which case it falls back to the heuristic):

```
Features per (query, candidate): [sketch sharpness, sketch roughness,
  window std, euclidean score, fourier score, nn score]
  + cap one-hot + sector one-hot
Model: 2-layer MLP → relevance logit (higher = better)
Train: python train_fusion.py
  - synthetic sketches = perturbed real windows (noise/smooth/time-warp)
  - ground truth: source window + same-symbol windows within 60 days
  - RankNet pairwise loss over (positive, negative) pairs
Ablation: held-out synthetic queries → P@5 / NDCG@5 (learned vs heuristic)
  and top-5 overlap, printed and stored in data/fusion_config.json
```

---

## Matcher Interface (Non-Negotiable)

```python
# matchers/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseMatcher(ABC):
    name: str  # shown in UI

    @abstractmethod
    def match(self, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
        """
        Args:
            sketch: z-normalized 50-point numpy array from user drawing
            candidates: list of window dicts (see Data Schema)
        Returns:
            same list with 'score' key added (lower = better match),
            sorted best-first
        """
        pass
```

**Every matcher must:**
- Accept exactly these two arguments
- Return the candidates list with `score` added
- Sort results best-first (lowest distance first)
- Never modify the input sketch or candidates in-place
- Never do I/O (no file reads, no network calls)

---

## Frontend Behaviour

### Canvas
- 600px wide, 200px tall
- Black background, white/purple draw line, 2.5px width
- On mousedown: start recording points
- On mousemove: draw line, append {x, y} to points array
- On mouseup: trigger `prepareAndSend()`

### `prepareAndSend()`
1. Extract y-values from points array
2. Resample to 50 values (linear interpolation in JS)
3. Z-normalize the 50 values
4. POST to `/search` with body:
```json
{
  "sketch": [50 floats],
  "filters": {
    "large": true,
    "mid": true,
    "small": false,
    "window_days": 60
  }
}
```
5. Open SSE stream to `/search-stream/{search_id}`
6. On each SSE event: update the progress UI

### Progress UI
Shows live as SSE events arrive:
```
Stage 1 — Hard filters         500 → 241  ✓
Stage 2 — Smoothed Euclidean   241 → 48   ✓
Stage 3 — Fourier              ranking... ✓
Converging results...                     ✓
```

### Results
5 cards rendered after stage 4 SSE event.
Each card:
- Symbol + company name
- Date window
- Match score (%)
- Matcher breakdown (Euclidean: X%, Fourier: Y%)
- Mini canvas chart: stock curve (green) + sketch overlay (purple dashed)

---

## API Endpoints

```
POST /search
  body: { sketch: float[], filters: {...} }
  returns: { search_id: str }   ← used to open SSE stream

GET /search-stream/{search_id}
  returns: text/event-stream
  streams: SSE events per stage (see SSE Event Schema)

GET /
  returns: static/index.html
```

---

## Normalization (Canonical Implementation)

Always use this exact implementation. Copy it, never rewrite it.

```python
def resample(arr: np.ndarray, target: int = 50) -> np.ndarray:
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target)
    return np.interp(x_new, x_old, arr)

def znorm(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mean) / std

def smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma)

def fingerprint(arr: np.ndarray, k: int = 8) -> np.ndarray:
    return np.abs(np.fft.fft(arr))[:k]

def sharpness(arr: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(arr, n=2))))

def sigmoid(x: float) -> float:
    import math
    return 1 / (1 + math.exp(-x))
```

---

## Requirements

```
fastapi
uvicorn
numpy
scipy
yfinance
pandas
python-multipart
```

---

## Nifty 500 Cap Categories

Use this mapping in `prepare_data.py`:

```python
LARGE_CAP = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
    "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS"
    # add remaining large caps
]

MID_CAP = [
    "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS", "OFSS.NS"
    # add mid caps
]

SMALL_CAP = [
    # add small caps
]
```

---

## What NOT To Build (Yet)

- No additional learned components without a research justification (no LLM/RAG,
  no heavy pretrained time-series models — the existing Siamese matcher and
  learned fusion layer cover the ML angle)
- No user accounts or saved searches  
- No real-time price data (historical only)
- No mobile-specific UI
- No database (just .npy file)
- No authentication
- No rate limiting
- No caching layer

These are v2+ features. Do not add them now.

---

## Research Hooks (Keep These In)

The following must be logged per search (to stdout is fine for now):
```python
{
  "search_id": str,
  "sharpness": float,
  "alpha": float,             # heuristic blend weight
  "blend_mode": "learned" | "heuristic",
  "heuristic_top": [symbols], # top-5 by heuristic sharpness blend
  "learned_top": [symbols],   # top-5 by learned fusion (same as heuristic if unused)
  "top5_overlap": float,      # 0..1 overlap of the two top-5 lists
  "stage1_count": int,
  "stage2_count": int,
  "stage3_count": int,
  "euclidean_top5": [symbols],
  "fourier_top5": [symbols],
  "final_top5": [symbols],
}
```

This data will eventually become the research paper findings.
Each search is one data point: did Fourier and Euclidean agree?
Did sharpness-based blending produce different results than either alone?

---

## Done Means

- [ ] Prepare data: `prepare_data.py` → `data/windows.npy` + `data/meta.json`
- [ ] Train Siamese matcher: `generate_pairs.py` + `train.py` → `data/siamese.pt`
- [ ] Train fusion ranker: `train_fusion.py` → `data/fusion.pt` + `data/fusion_config.json`
- [ ] `python main.py` starts without errors
- [ ] Canvas draws smoothly on mouse drag
- [ ] Drawing triggers POST and opens SSE stream
- [ ] All 4 SSE stage events fire in order
- [ ] Progress UI updates live per stage
- [ ] 5 result cards render with mini charts
- [ ] Sketch overlay visible on each result card
- [ ] Cap filter checkboxes work (toggle large/mid/small)
- [ ] Window length dropdown works (30/60/90 days)
- [ ] Research log prints to stdout per search (incl. `blend_mode`, `heuristic_top`, `learned_top`)

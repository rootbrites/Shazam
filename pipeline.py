"""
pipeline.py — Three-stage search pipeline with SSE streaming.

Stage 1: Hard filters (cap + std thresholds)
Stage 2: Smoothed Euclidean (via EuclideanMatcher)
Stage 3: Fourier matching (via FourierMatcher)
Convergence: sharpness-weighted blend → top 5 results

Each stage yields an SSE event dict. The caller (main.py) is responsible
for serializing and streaming these to the frontend.
"""

import math
import json
import numpy as np
from typing import AsyncGenerator, Optional

from matchers.euclidean import EuclideanMatcher
from matchers.fourier   import FourierMatcher
from matchers.base      import vertical_mae, mae_to_pct, shape_score

# NeuralMatcher is optional — only available after python train.py
try:
    from matchers.nn import NeuralMatcher as _NeuralMatcher
    _neural_matcher = _NeuralMatcher()
except FileNotFoundError:
    _neural_matcher = None

# Learned fusion layer is optional — only available after python train_fusion.py
try:
    from fusion import load_fusion, make_feature_vector
    _fusion = load_fusion()
except Exception:  # noqa: BLE001 — degrade gracefully to the heuristic blend
    _fusion = None


# ---------------------------------------------------------------------------
# Canonical normalization helpers (from AGENTS.md — copied verbatim)
# ---------------------------------------------------------------------------

def resample(arr: np.ndarray, target: int = 50) -> np.ndarray:
    """Resample arr to `target` points using linear interpolation."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target)
    return np.interp(x_new, x_old, arr)


def znorm(arr: np.ndarray) -> np.ndarray:
    """Z-normalize arr to zero mean and unit std (epsilon prevents div-by-zero)."""
    mean = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mean) / std


def smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing with given sigma."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma)


def fingerprint(arr: np.ndarray, k: int = 8) -> np.ndarray:
    """Return first k magnitudes of the FFT of arr."""
    return np.abs(np.fft.fft(arr))[:k]


def sharpness(arr: np.ndarray) -> float:
    """Compute second-order difference mean as a sharpness metric."""
    return float(np.mean(np.abs(np.diff(arr, n=2))))


def sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------------------
# Stage 1 — Hard filters
# ---------------------------------------------------------------------------

def stage1_filter(
    windows: list[dict],
    filters: dict,
    std_thresh_low: float = 0.01,
    std_thresh_high: float = 3.0,
) -> list[dict]:
    """
    Apply hard filters to the full window list.

    Filters by cap category (large/mid/small booleans) and std thresholds
    to remove flat or pure-noise windows.
    """
    cap_filter = {
        "large": filters.get("large", True),
        "mid":   filters.get("mid", True),
        "small": filters.get("small", False),
    }

    candidates = []
    for w in windows:
        # Cap filter
        if not cap_filter.get(w["cap"], False):
            continue
        # Std filter: kick flat and pure-noise windows
        if not (std_thresh_low < w["std"] < std_thresh_high):
            continue
        candidates.append(w)

    return candidates


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def _enrich(cand: dict, score: float, meta_map: dict, sketch: np.ndarray,
            scoring_mode: str = "abs_var") -> dict:
    """Build a result dict, computing the ranking score based on scoring_mode."""
    symbol = cand["symbol"]
    meta   = meta_map.get(symbol, {})
    eu = cand.get("euclidean_score", 0.0)
    fo = cand.get("fourier_score",   0.0)
    nn = cand.get("nn_score")
    matcher_scores = {"Euclidean": round(eu, 6), "Fourier": round(fo, 6)}
    if nn is not None:
        matcher_scores["Neural Net"] = round(nn, 6)

    errors_abs    = np.abs(sketch - cand["norm"])          # |e_i|
    errors_signed = sketch - cand["norm"]                   # e_i (signed)
    mae           = float(np.mean(errors_abs))
    std_abs       = float(np.std(errors_abs))               # std(|e_i|)
    std_signed    = float(np.std(errors_signed))            # std(e_i)

    if scoring_mode == "mae":
        final_score = mae
    elif scoring_mode == "signed_var":
        final_score = mae + std_signed
    else:                                                    # "abs_var" (default)
        final_score = mae + std_abs

    return {
        "symbol":         symbol,
        "name":           meta.get("name", symbol),
        "date_end":       cand["date_end"],
        "score":          round(final_score, 6),
        "mae":            round(mae, 4),
        "std_abs":        round(std_abs, 4),
        "std_signed":     round(std_signed, 4),
        "scoring_mode":   scoring_mode,
        "match_pct":      mae_to_pct(mae),
        "curve":          cand["norm"].tolist(),
        "raw_curve":      cand["raw"].tolist(),
        "matcher_scores": matcher_scores,
    }


def _top_n_deduped(
    scored: list[tuple[float, dict]],
    n: int,
    meta_map: dict,
    sketch: np.ndarray,
    scoring_mode: str = "abs_var",
    reverse: bool = False,
) -> list[dict]:
    """
    Use matcher score to select candidates (dedup by symbol), then re-sort
    the collected top-n by the active scoring_mode ascending.
    """
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=reverse)
    seen: set[str] = set()
    collected: list[dict] = []
    for matcher_score, cand in scored_sorted:
        sym = cand["symbol"]
        if sym not in seen:
            seen.add(sym)
            collected.append(_enrich(cand, matcher_score, meta_map, sketch, scoring_mode))
        if len(collected) == n:
            break
    collected.sort(key=lambda r: r["score"])
    return collected


def build_per_matcher_results(
    candidates: list[dict],
    sketch: np.ndarray,
    meta_map: dict[str, dict],
    enabled: set[str],
    scoring_mode: str = "abs_var",
    n: int = 5,
    fusion: Optional[tuple] = None,
) -> dict[str, list[dict]]:
    """
    Build independent top-n lists for each enabled matcher plus a Blended list.

    Returns a dict keyed by matcher name:
      { "Euclidean": [...], "Fourier": [...], "Neural Net": [...], "Blended": [...] }
    Disabled matchers map to an empty list. Each result has mae, std_abs, std_signed.
    scoring_mode controls the final ranking: 'mae' | 'abs_var' | 'signed_var'.
    When fusion=(ranker, spec) is provided, the Blended list is selected with
    the learned fusion score instead of the heuristic sharpness blend.
    """
    sharp = sharpness(sketch)
    alpha = sigmoid(sharp * 5)
    ranker, spec = fusion if fusion is not None else (None, None)

    eu_scored: list[tuple[float, dict]] = []
    fo_scored: list[tuple[float, dict]] = []
    nn_scored: list[tuple[float, dict]] = []
    bl_scored: list[tuple[float, dict]] = []

    for cand in candidates:
        eu = cand.get("euclidean_score", 0.0)
        fo = cand.get("fourier_score",   0.0)
        nn = cand.get("nn_score")

        if "Euclidean" in enabled:
            eu_scored.append((eu, cand))
        if "Fourier" in enabled:
            fo_scored.append((fo, cand))
        if "Neural Net" in enabled and nn is not None:
            nn_scored.append((nn, cand))

        if ranker is not None:
            prob = float(ranker.predict_proba(
                make_feature_vector(sketch, cand, spec)[None, :])[0])
            bl_scored.append((prob, cand))
        else:
            eu_term = (eu + nn) / 2.0 if (nn is not None and "Neural Net" in enabled) else eu
            final   = alpha * eu_term + (1 - alpha) * fo
            bl_scored.append((final, cand))

    kw = dict(scoring_mode=scoring_mode)
    return {
        "Euclidean":  _top_n_deduped(eu_scored, n, meta_map, sketch, **kw) if eu_scored else [],
        "Fourier":    _top_n_deduped(fo_scored, n, meta_map, sketch, **kw) if fo_scored else [],
        "Neural Net": _top_n_deduped(nn_scored, n, meta_map, sketch, **kw) if nn_scored else [],
        "Blended":    _top_n_deduped(bl_scored, n, meta_map, sketch, reverse=(ranker is not None), **kw),
    }


def _top_symbols(scored: list[tuple[float, str]], reverse: bool = False) -> list[str]:
    """Best-first deduped top-5 symbol list for the research log."""
    seen: set[str] = set()
    out: list[str] = []
    for score, sym in sorted(scored, key=lambda x: x[0], reverse=reverse):
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
        if len(out) == 5:
            break
    return out


# ---------------------------------------------------------------------------
# Main pipeline — async generator yielding SSE event dicts
# ---------------------------------------------------------------------------

async def run_pipeline(
    sketch: np.ndarray,
    filters: dict,
    all_windows: list[dict],
    meta_map: dict[str, dict],
    search_id: str,
) -> AsyncGenerator[dict, None]:
    """
    Run the full 3-stage search pipeline and yield SSE event dicts.

    Yields one dict per stage. The caller must serialize to SSE format.

    Args:
        sketch:      z-normalized 50-point array from the user's drawing
        filters:     dict with keys large/mid/small (bools) and window_days (int)
        all_windows: preloaded list of window dicts from data/windows.npy
        meta_map:    dict mapping symbol -> metadata dict
        search_id:   unique search ID for research logging
    """
    total = len(all_windows)
    enabled_matchers: set[str] = set(filters.get("matchers", ["Euclidean", "Fourier", "Neural Net"]))
    scoring_mode: str = filters.get("scoring_mode", "abs_var")

    # ------------------------------------------------------------------
    # Stage 1 — Hard filters
    # ------------------------------------------------------------------
    candidates = stage1_filter(all_windows, filters)
    stage1_count = len(candidates)

    yield {
        "stage": 1,
        "label": "Hard filters",
        "status": "done",
        "remaining": stage1_count,
        "total": total,
        "results": [],
    }

    # ------------------------------------------------------------------
    # Stage 2 — Smoothed Euclidean
    # ------------------------------------------------------------------
    smoothing = filters.get("smoothing", 2.0)
    euclidean_matcher = EuclideanMatcher(sigma=smoothing)
    candidates = euclidean_matcher.match(sketch, candidates)
    stage2_count = len(candidates)

    yield {
        "stage": 2,
        "label": "Smoothed Euclidean",
        "status": "done",
        "remaining": stage2_count,
        "total": total,
        "results": [],
    }

    # ------------------------------------------------------------------
    # Stage 3a — Fourier
    # ------------------------------------------------------------------
    fourier_matcher = FourierMatcher()
    candidates = fourier_matcher.match(sketch, candidates)
    stage3_count = len(candidates)

    yield {
        "stage": 3,
        "label": "Fourier",
        "status": "done",
        "remaining": stage3_count,
        "total": total,
        "results": [],
    }

    # ------------------------------------------------------------------
    # Stage 3b — Neural Net (optional; skipped if disabled or not loaded)
    # ------------------------------------------------------------------
    if _neural_matcher is not None and "Neural Net" in enabled_matchers:
        nn_results = _neural_matcher.match(sketch, candidates)
        nn_score_map = {
            (r["symbol"], r["date_end"]): r["score"]
            for r in nn_results
        }
        candidates = [
            {**c, "nn_score": nn_score_map.get((c["symbol"], c["date_end"]), None)}
            for c in candidates
        ]

    # ------------------------------------------------------------------
    # Convergence layer — build per-matcher + blended top-5 lists
    # ------------------------------------------------------------------
    sharp = sharpness(sketch)
    alpha = sigmoid(sharp * 5)

    nn_active = (_neural_matcher is not None and "Neural Net" in enabled_matchers)
    fusion_active = (
        _fusion is not None
        and bool(_fusion[1].get("nn_used", False)) == bool(nn_active)
    )

    per_matcher = build_per_matcher_results(
        candidates, sketch, meta_map, enabled_matchers,
        scoring_mode=scoring_mode, n=5,
        fusion=_fusion if fusion_active else None,
    )
    blended_top = per_matcher["Blended"]

    # Research-log top lists: heuristic sharpness blend vs learned fusion
    heuristic_scored = []
    for c in candidates:
        eu = c.get("euclidean_score", 9e9)
        fo = c.get("fourier_score", 9e9)
        nn = c.get("nn_score")
        eu_term = (eu + nn) / 2.0 if (nn is not None and nn_active) else eu
        heuristic_scored.append((alpha * eu_term + (1 - alpha) * fo, c["symbol"]))
    heuristic_top = _top_symbols(heuristic_scored, reverse=False)
    learned_top = heuristic_top
    if fusion_active:
        ranker, spec = _fusion
        learned_scored = []
        for c in candidates:
            prob = float(ranker.predict_proba(
                make_feature_vector(sketch, c, spec)[None, :])[0])
            learned_scored.append((prob, c["symbol"]))
        learned_top = _top_symbols(learned_scored, reverse=True)

    # Research log (stdout as required by AGENTS.md)
    euclidean_top = [c["symbol"] for c in sorted(
        candidates, key=lambda x: x.get("euclidean_score", 9e9))[:10]]
    fourier_top   = [c["symbol"] for c in sorted(
        candidates, key=lambda x: x.get("fourier_score", 9e9))[:10]]
    nn_top        = [c["symbol"] for c in sorted(
        candidates, key=lambda x: x.get("nn_score") or 9e9)[:10]] \
        if nn_active else []

    research_log = {
        "search_id":    search_id,
        "sharpness":    round(sharp, 6),
        "alpha":        round(alpha, 6),
        "blend_mode":   "learned" if fusion_active else "heuristic",
        "heuristic_top": heuristic_top,
        "learned_top":  learned_top,
        "top5_overlap": round(len(set(heuristic_top) & set(learned_top)) / 5.0, 2),
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "stage3_count": stage3_count,
        "enabled_matchers": list(enabled_matchers),
        "euclidean_top": euclidean_top,
        "fourier_top":   fourier_top,
        "nn_top":        nn_top,
        "final_top":     [r["symbol"] for r in blended_top],
    }
    print(json.dumps(research_log))

    # Calculate smoothed sketch for overlay
    sigma = filters.get("smoothing", 2.0)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed_sketch = gaussian_filter1d(sketch.astype(np.float64), sigma=sigma)
    else:
        smoothed_sketch = sketch.astype(np.float64)

    yield {
        "stage": 4,
        "label": "Final results",
        "status": "done",
        "remaining": len(blended_top),
        "total": total,
        "results": per_matcher,          # dict keyed by matcher name
        "smoothed_sketch": smoothed_sketch.tolist(),
    }

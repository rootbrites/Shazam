"""
fusion.py — Learned fusion layer: ranks candidates by combining all matcher scores.

Trains a small MLP that replaces the hand-tuned sharpness-weighted blend
(alpha) in pipeline.py. Features per (query, candidate) pair: sketch shape
stats, matcher scores, and window metadata (std, cap, sector).

Training data needs no human labels — synthetic sketches are created by
perturbing real windows, and the source window plus its same-symbol
temporal neighbours are the known-relevant ground truth (see
train_fusion.py). This module provides the model, feature extraction,
and synthetic-query generation.
"""

import json
import os
from typing import Callable, Optional

import numpy as np
import torch

from matchers.siamese import SiameseNet

FUSION_PATH = os.path.join("data", "fusion.pt")
FUSION_CONFIG_PATH = os.path.join("data", "fusion_config.json")

CAP_ORDER = ["large", "mid", "small"]

# Base feature keys — order must be identical between train and inference.
BASE_FEATURES = [
    "sketch_sharpness",   # 2nd-difference energy of the sketch
    "sketch_roughness",   # 1st-difference energy of the sketch
    "window_std",         # raw std of the candidate window
    "euclidean",          # Stage 2 smoothed-Euclidean distance
    "fourier",            # Stage 3a FFT-magnitude distance
    "nn",                 # Stage 3b embedding L2 distance (0.0 if unused)
]


# ---------------------------------------------------------------------------
# Canonical normalization helpers (copied verbatim from AGENTS.md)
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


# ---------------------------------------------------------------------------
# Sketch statistics
# ---------------------------------------------------------------------------

def sketch_sharpness(arr: np.ndarray) -> float:
    """Second-order difference mean — the same sharpness metric as pipeline.py."""
    return float(np.mean(np.abs(np.diff(arr, n=2))))


def sketch_roughness(arr: np.ndarray) -> float:
    """First-order difference mean — captures how jagged the sketch is."""
    return float(np.mean(np.abs(np.diff(arr))))


# ---------------------------------------------------------------------------
# Synthetic query generation
# ---------------------------------------------------------------------------

def synthesize_sketches(window: dict, rng: np.random.Generator, n: int = 2) -> list[np.ndarray]:
    """
    Make n synthetic user sketches by perturbing a window's normalized curve.

    Perturbations are deliberately harsh (mimicking rough hand drawings):
    noise 0.3-0.7 sigma, smoothing sigma 0-3, and time-warps up to 4% of the
    window. Variants rotate: (0) noise only, (1) resmoothing then noise,
    (2) time-warp then noise. Each result is z-normalized to 50 points.
    """
    base = np.asarray(window["norm"], dtype=np.float64)
    sketches: list[np.ndarray] = []
    for i in range(n):
        x = base
        if i % 3 == 1:
            from scipy.ndimage import gaussian_filter1d
            x = gaussian_filter1d(x, sigma=float(rng.uniform(0.0, 3.0)))
        elif i % 3 == 2:
            t_old = np.linspace(0.0, 1.0, len(base))
            t_new = np.clip(t_old + rng.normal(0.0, 0.04, len(base)), 0.0, 1.0)
            t_new[0], t_new[-1] = 0.0, 1.0
            x = np.interp(t_new, t_old, base)
        x = x + rng.normal(0.0, float(rng.uniform(0.30, 0.70)), len(x))
        sketches.append(znorm(x))
    return sketches


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def build_spec(windows: list[dict]) -> dict:
    """Build the feature spec (base features + cap/sector ordering) from window data."""
    sectors = sorted({str(w.get("sector", "Unknown")) for w in windows})
    return {
        "base_features": BASE_FEATURES,
        "cap_order": CAP_ORDER,
        "sector_order": sectors,
        "nn_used": None,   # set by the trainer once NN availability is known
        "input_dim": 0,    # filled by make_feature_vector
    }


def make_feature_vector(sketch: np.ndarray, cand: dict, spec: dict) -> np.ndarray:
    """Build the fusion feature vector for one (query, candidate) pair."""
    vec: list[float] = [
        sketch_sharpness(sketch),
        sketch_roughness(sketch),
        float(cand.get("std", 0.0)),
        float(cand.get("euclidean_score", 0.0)),
        float(cand.get("fourier_score", 0.0)),
        float(cand.get("nn_score", 0.0)),
    ]
    vec += [1.0 if cand.get("cap") == c else 0.0 for c in spec["cap_order"]]
    vec += [1.0 if cand.get("sector") == s else 0.0 for s in spec["sector_order"]]
    return np.asarray(vec, dtype=np.float32)


def feature_dim(spec: dict) -> int:
    """Total feature dimensionality for a spec."""
    return len(spec["base_features"]) + len(spec["cap_order"]) + len(spec["sector_order"])


# ---------------------------------------------------------------------------
# Siamese embedder (reused by training for the NN feature)
# ---------------------------------------------------------------------------

def siamese_embedder() -> Optional[tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]]:
    """
    Return (embed_fn, window_embeddings) for the trained SiameseNet, or None.

    This matches matchers/nn.NeuralMatcher: embedding L2 distance is the NN
    score. window_embeddings is (N, 64) for whatever curves are passed in
    by the caller (training embeds the full window list once).
    """
    path = os.path.join("data", "siamese.pt")
    if not os.path.exists(path):
        return None
    model = SiameseNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    def embed(curves: np.ndarray) -> np.ndarray:
        """Batch-embed an (N, 50) curve array to an (N, 64) embedding matrix."""
        t = torch.as_tensor(np.asarray(curves, dtype=np.float32)).unsqueeze(1)
        with torch.no_grad():
            return model.encode(t).numpy()

    return embed, None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FusionRanker(torch.nn.Module):
    """Small MLP: fusion features → relevance logit (higher = better match)."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32),
                 dropout: float = 0.1):
        """Build Linear → ReLU → Dropout blocks followed by a single logit."""
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_dim) → (batch,) logits."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Sigmoid relevance probabilities for an (N, D) feature matrix."""
        x = torch.as_tensor(features, dtype=torch.float32)
        return torch.sigmoid(self(x)).numpy()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_fusion(ranker: FusionRanker, config: dict,
                weights_path: str = FUSION_PATH,
                config_path: str = FUSION_CONFIG_PATH) -> None:
    """Save model weights and its feature spec/config as JSON."""
    torch.save(ranker.state_dict(), weights_path)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"[fusion] saved -> {weights_path} + {config_path}")


def load_fusion(weights_path: str = FUSION_PATH,
                config_path: str = FUSION_CONFIG_PATH) -> Optional[tuple[FusionRanker, dict]]:
    """Load (ranker, config) from disk, or return None if absent/unloadable."""
    if not (os.path.exists(weights_path) and os.path.exists(config_path)):
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        ranker = FusionRanker(config["input_dim"], tuple(config["hidden_dims"]))
        ranker.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        ranker.eval()
        return ranker, config
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to heuristic blend
        print(f"[fusion] load failed, falling back to heuristic: {exc}")
        return None

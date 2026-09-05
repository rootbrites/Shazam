"""
train_fusion.py — Train the learned fusion layer (RankNet-style MLP).

Pipeline:
  1. Load data/windows.npy (also data/siamese.pt if present).
  2. Generate synthetic queries: perturb real windows into sketches.
     Ground truth: the source window + same-symbol windows within DATE_WINDOW
     days are relevant; everything else is negative.
  3. Per query, draw a candidate pool and run the real matcher chain
     (Euclidean top-20% → Fourier → Neural) so training features are
     identical to inference-time features in pipeline.py.
  4. Train a small MLP with pairwise ranking loss (RankNet).
  5. Ablation on held-out queries: learned fusion vs the heuristic
     sharpness blend → P@5 / NDCG@5 / overlap, saved with the model.

Usage:
    python train_fusion.py                # full run (~150 queries)
    python train_fusion.py --queries 20   # quick smoke test
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from fusion import (
    FusionRanker,
    build_spec,
    feature_dim,
    make_feature_vector,
    save_fusion,
    siamese_embedder,
    synthesize_sketches,
)
from matchers.euclidean import EuclideanMatcher
from matchers.fourier import FourierMatcher
from pipeline import sharpness, sigmoid

WINDOWS_PATH = os.path.join("data", "windows.npy")
DATE_WINDOW = 60           # days — same-symbol temporal-neighbour relevance
POOL_SIZE = 1500           # candidate pool sampled per query (pre stage-2)
VAL_FRAC = 0.2             # fraction of queries held out for the ablation
LR = 1e-3
EPOCHS = 12
BATCH = 256
MAX_PAIRS_PER_QUERY = 400  # cap positive×negative pairs per query
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.1


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_windows() -> list[dict]:
    """Load the precomputed window list from data/windows.npy."""
    with open(WINDOWS_PATH, "rb") as f:
        return list(np.load(f, allow_pickle=True))


def _parse_date(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD date_end string."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def is_relevant(cand: dict, src: dict) -> bool:
    """True when cand is the source window or a same-symbol temporal neighbour."""
    if cand["symbol"] != src["symbol"]:
        return False
    gap = abs((_parse_date(cand["date_end"]) - _parse_date(src["date_end"])).days)
    return gap <= DATE_WINDOW


def sample_pool(windows: list[dict], src: dict, rng: np.random.Generator,
                pool_size: int = POOL_SIZE) -> list[dict]:
    """
    Build a query's candidate pool: all relevant windows plus random fillers.

    Entries are shallow copies carrying the global index in '_idx' so the
    precomputed Siamese embeddings can be looked up after the matcher chain.
    """
    relevant = [i for i, w in enumerate(windows) if is_relevant(w, src)]
    others = [i for i in range(len(windows)) if i not in relevant]
    rng.shuffle(others)
    chosen = relevant + others[: max(0, pool_size - len(relevant))]
    return [{**windows[i], "_idx": i} for i in chosen]


def matcher_chain(sketch: np.ndarray, pool: list[dict],
                  euclid: EuclideanMatcher, fourier: FourierMatcher,
                  embed_fn, embeddings: Optional[np.ndarray]) -> list[dict]:
    """
    Replicate the pipeline matcher chain; return stage-3 survivors with
    euclidean_score, fourier_score and (optionally) nn_score keys set.
    """
    survivors = euclid.match(sketch, pool)          # Stage 2: top-20%
    survivors = fourier.match(sketch, survivors)    # Stage 3a
    if embed_fn is not None and embeddings is not None:
        sketch_emb = embed_fn(sketch[None, :])[:, 0]           # (64,)
        idx = np.array([c["_idx"] for c in survivors])
        dists = np.linalg.norm(embeddings[idx] - sketch_emb, axis=1)
        survivors = [
            {**c, "nn_score": float(d)} for c, d in zip(survivors, dists)
        ]
    return survivors


def build_query_rows(sketch: np.ndarray, survivors: list[dict],
                     src: dict) -> list[tuple]:
    """
    Convert stage-3 survivors into training rows.

    Each row: (features, label, symbol, euclidean, fourier, nn).
    Label = 1 for the source window and its same-symbol neighbours.
    """
    rows: list[tuple] = []
    for c in survivors:
        label = 1.0 if is_relevant(c, src) else 0.0
        rows.append((
            make_feature_vector(sketch, c, _SPEC),
            label,
            c["symbol"],
            float(c["euclidean_score"]),
            float(c["fourier_score"]),
            float(c.get("nn_score", 0.0)),
        ))
    return rows


# Global spec shared by feature extraction (set in main).
_SPEC: dict = {}


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _dedup_top(scored: list[tuple[float, str]], n: int = 5,
               reverse: bool = False) -> list[str]:
    """Best-first top-n symbols, dropping duplicate symbols."""
    seen: set[str] = set()
    out: list[str] = []
    for score, sym in sorted(scored, key=lambda x: x[0], reverse=reverse):
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
        if len(out) == n:
            break
    return out


def ndcg_at5(ranking: list[str], relevant: set[str]) -> float:
    """Binary-gain NDCG@5 for a symbol ranking against the relevant set."""
    dcg = sum(1.0 / np.log2(i + 2) for i, s in enumerate(ranking[:5]) if s in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(5, len(relevant))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at5(ranking: list[str], relevant: set[str]) -> float:
    """Fraction of the top-5 that is relevant."""
    return sum(1 for s in ranking[:5] if s in relevant) / 5.0


def recall_at5(ranking: list[str], relevant: set[str]) -> float:
    """Fraction of relevant symbols found in the top-5."""
    return sum(1 for s in ranking[:5] if s in relevant) / max(1, len(relevant))


def eval_query(rows: list[tuple], sketch: np.ndarray, src: dict,
               use_fusion: bool, ranker: Optional[FusionRanker] = None) -> tuple[list, list]:
    """
    Rank a query's rows with fusion (if use_fusion) or the heuristic blend.

    Returns (ranking_list, heuristic_list) of deduped top-5 symbol lists.
    """
    relevant = {r[2] for r in rows if r[1] == 1.0}
    scores = []
    for feats, label, sym, eu, fo, nn in rows:
        if use_fusion and ranker is not None:
            s = float(ranker.predict_proba(feats[None, :])[0])
            scores.append((s, sym, eu, fo, nn))
        else:
            alpha = sigmoid(sharpness(sketch) * 5.0)
            eu_term = (eu + nn) / 2.0 if nn > 0 else eu
            scores.append((alpha * eu_term + (1 - alpha) * fo, sym, eu, fo, nn))
    if use_fusion and ranker is not None:
        learned = _dedup_top([(s, sym) for s, sym, _, _, _ in scores], 5, reverse=True)
        heuristic = _dedup_top(
            [(alpha_heuristic(sketch, eu, fo, nn), sym) for _, sym, eu, fo, nn in scores],
            5, reverse=False)
    else:
        learned = _dedup_top([(s, sym) for s, sym, _, _, _ in scores], 5, reverse=False)
        heuristic = learned
    return learned, heuristic


def alpha_heuristic(sketch: np.ndarray, eu: float, fo: float, nn: float) -> float:
    """The pipeline's heuristic blend score for one candidate."""
    alpha = sigmoid(sharpness(sketch) * 5.0)
    eu_term = (eu + nn) / 2.0 if nn > 0 else eu
    return alpha * eu_term + (1 - alpha) * fo


# ---------------------------------------------------------------------------
# Pairwise training
# ---------------------------------------------------------------------------

def make_pairs(query_rows: list[tuple], rng: random.Random,
               max_pairs: int = MAX_PAIRS_PER_QUERY) -> list[tuple[np.ndarray, np.ndarray]]:
    """RankNet-style (positive, negative) feature pairs for one query."""
    pos = [r[0] for r in query_rows if r[1] == 1.0]
    neg = [r[0] for r in query_rows if r[1] == 0.0]
    if not pos or not neg:
        return []
    pairs = [(p, n) for p in pos for n in neg]
    rng.shuffle(pairs)
    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate queries, train the ranker, run the ablation, save the model."""
    parser = argparse.ArgumentParser(description="Train the learned fusion layer")
    parser.add_argument("--queries", type=int, default=150, help="synthetic queries to generate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE, help="candidates per query")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {WINDOWS_PATH} ...")
    windows = load_windows()
    global _SPEC
    _SPEC = build_spec(windows)
    nn_used = os.path.exists(os.path.join("data", "siamese.pt"))
    _SPEC["nn_used"] = nn_used

    embed_fn = embeddings = None
    if nn_used:
        embed_fn, _ = siamese_embedder()
        norms = np.stack([w["norm"].astype(np.float32) for w in windows])
        embeddings = embed_fn(norms)
        print(f"[fusion] NN features on — embedded {len(windows)} windows")

    # --- Generate queries -------------------------------------------------
    euclid = EuclideanMatcher(sigma=2.0)
    fourier = FourierMatcher()
    query_ids = list(rng.choice(len(windows), size=args.queries, replace=False))
    rows_by_qid: dict[int, list[tuple]] = {}
    sketch_by_qid: dict[int, np.ndarray] = {}
    src_by_qid: dict[int, dict] = {}
    n_skipped = 0

    for qi in query_ids:
        src = windows[qi]
        sketch = synthesize_sketches(src, rng, n=1)[0]
        pool = sample_pool(windows, src, rng, args.pool_size)
        survivors = matcher_chain(sketch, pool, euclid, fourier, embed_fn, embeddings)
        rows = build_query_rows(sketch, survivors, src)
        if not any(r[1] == 1.0 for r in rows) or not any(r[1] == 0.0 for r in rows):
            n_skipped += 1
            continue
        rows_by_qid[qi] = rows
        sketch_by_qid[qi] = sketch
        src_by_qid[qi] = src

    qids = list(rows_by_qid.keys())
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * VAL_FRAC))
    val_qids, train_qids = qids[:n_val], qids[n_val:]
    print(f"[fusion] {len(qids)} usable queries ({n_skipped} skipped) "
          f"→ {len(train_qids)} train / {len(val_qids)} val")

    # --- Train ------------------------------------------------------------
    dim = feature_dim(_SPEC)
    ranker = FusionRanker(dim, HIDDEN_DIMS, dropout=DROPOUT)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=LR)

    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for qi in train_qids:
        all_pairs += make_pairs(rows_by_qid[qi], random.Random(args.seed + qi))
    print(f"[fusion] {len(all_pairs)} ranking pairs")

    best_ndcg, best_state = -1.0, None
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(all_pairs)
        ranker.train()
        epoch_loss = 0.0
        for i in range(0, len(all_pairs), BATCH):
            batch = all_pairs[i:i + BATCH]
            p = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
            n = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
            loss = F.softplus(-(ranker(p) - ranker(n))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= max(1, len(all_pairs))

        ranker.eval()
        learned_top5, heuristic_top5 = [], []
        for qi in val_qids:
            l, h = eval_query(rows_by_qid[qi], sketch_by_qid[qi], src_by_qid[qi],
                              use_fusion=True, ranker=ranker)
            learned_top5.append(l)
            heuristic_top5.append(h)
        rel = [{r[2] for r in rows_by_qid[qi] if r[1] == 1.0} for qi in val_qids]
        prec_l = np.mean([precision_at5(l, rel[i]) for i, l in enumerate(learned_top5)])
        prec_h = np.mean([precision_at5(h, rel[i]) for i, h in enumerate(heuristic_top5)])
        rec_l = np.mean([recall_at5(l, rel[i]) for i, l in enumerate(learned_top5)])
        rec_h = np.mean([recall_at5(h, rel[i]) for i, h in enumerate(heuristic_top5)])
        ndcg_l = np.mean([ndcg_at5(l, rel[i]) for i, l in enumerate(learned_top5)])
        ndcg_h = np.mean([ndcg_at5(h, rel[i]) for i, h in enumerate(heuristic_top5)])
        print(f"epoch {epoch + 1:2d}  loss={epoch_loss:.4f}  "
              f"P@5: learned={prec_l:.3f} heuristic={prec_h:.3f}  "
              f"R@5: learned={rec_l:.3f} heuristic={rec_h:.3f}  "
              f"NDCG@5: learned={ndcg_l:.3f} heuristic={ndcg_h:.3f}")
        if ndcg_l > best_ndcg:
            best_ndcg = ndcg_l
            best_state = {k: v.clone() for k, v in ranker.state_dict().items()}

    if best_state is not None:
        ranker.load_state_dict(best_state)
    ranker.eval()

    # --- Final ablation with the best weights -----------------------------
    learned_top5, heuristic_top5 = [], []
    for qi in val_qids:
        l, h = eval_query(rows_by_qid[qi], sketch_by_qid[qi], src_by_qid[qi],
                          use_fusion=True, ranker=ranker)
        learned_top5.append(l)
        heuristic_top5.append(h)
    rel = [{r[2] for r in rows_by_qid[qi] if r[1] == 1.0} for qi in val_qids]
    metrics = {
        "val_queries": len(val_qids),
        "precision5_learned": float(np.mean(
            [precision_at5(l, rel[i]) for i, l in enumerate(learned_top5)])),
        "precision5_heuristic": float(np.mean(
            [precision_at5(h, rel[i]) for i, h in enumerate(heuristic_top5)])),
        "recall5_learned": float(np.mean(
            [recall_at5(l, rel[i]) for i, l in enumerate(learned_top5)])),
        "recall5_heuristic": float(np.mean(
            [recall_at5(h, rel[i]) for i, h in enumerate(heuristic_top5)])),
        "ndcg5_learned": float(np.mean(
            [ndcg_at5(l, rel[i]) for i, l in enumerate(learned_top5)])),
        "ndcg5_heuristic": float(np.mean(
            [ndcg_at5(h, rel[i]) for i, h in enumerate(heuristic_top5)])),
        "top5_overlap": float(np.mean(
            [len(set(l) & set(h)) / 5.0 for l, h in zip(learned_top5, heuristic_top5)])),
    }
    print("\n=== Ablation (held-out synthetic queries) ===")
    for k, v in metrics.items():
        print(f"  {k:16s} {v:.4f}")

    config = {
        "feature_names": _SPEC["base_features"],
        "cap_order": _SPEC["cap_order"],
        "sector_order": _SPEC["sector_order"],
        "nn_used": nn_used,
        "input_dim": dim,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": DROPOUT,
        "queries": args.queries,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    save_fusion(ranker, config)


if __name__ == "__main__":
    main()

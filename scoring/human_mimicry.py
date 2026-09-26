"""Human mimicry: can a discriminator tell model vectors from rater vectors?

Frozen definition (SPEC.md section 3, v2.0.0):

1. Rows. For every holdout item that has both human rater rows
   (`annotation/rater_vectors.jsonl`) and a model prediction (the `pred`
   vector in a results file's `per_item`), build a design matrix. Each
   human row is centered by that item's **leave-one-rater-out consensus**
   (per-dimension mean across the *other* raters - the same principle as
   the human ceiling: a rater is never measured against a mean they helped
   create). The model row is centered by the full consensus, which the
   model by definition did not contribute to. Centering removes item
   identity and consensus level, so the discriminator can only see rating
   *shape*: how far from consensus a rating sits, and in which dimensions.
   A model that echoes the consensus exactly is trivially separable,
   because no human rater ever deviates by zero on all 17 dims.

2. Discriminator. Gaussian discriminant with diagonal covariance (closed
   form, no hyperparameters, no randomness): per class, the per-dimension
   mean and variance of the centered rows; a row is scored by the Gaussian
   log-likelihood ratio model-vs-human. Diagonal, not full covariance,
   because the model class has ~84 rows and a 17x17 covariance would be
   noise. Unlike a mean-only (Fisher LDA) rule this also fires on
   *variance* differences - a model whose deviations are smeared toward
   consensus with zero-mean noise is caught, not waved through. Trained
   leave-one-item-out: the held-out item's rows are scored by a
   discriminator that never saw them.

3. Score. Pool out-of-fold scores and compute the tie-averaged
   Mann-Whitney AUC (positive class = model). Then

       human_mimicry = 2 * (1 - max(AUC, 1 - AUC))   in [0, 1]

   1.0 = discriminator at chance: the model's deviations from consensus
   are indistinguishable from a human rater's. 0.0 = fully separable.

Reading the number. 1.0 is stricter than any actual rater achieves: run
the same protocol with each real rater as the "model"
(`human_reference`, `--reference`) and the discriminator learns that
rater's fingerprint. Publish the reference band next to the score; a
model inside the band is behaviorally exchangeable with an individual
rater even though it is not at chance.

Returns None (not_measured) when fewer than MIN_ITEMS shared items have
both classes. Roadmap slot: reported per model, never averaged into a
headline (score.ROADMAP_KEYS).

    python scoring/human_mimicry.py                 # all results/*.json
    python scoring/human_mimicry.py results/gpt-5-4.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_ceiling
import score

REPO = Path(__file__).resolve().parents[1]
RATER_FILE = REPO / "annotation" / "rater_vectors.jsonl"
MIN_ITEMS = 8
VAR_FLOOR = 1e-3


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Tie-averaged 1-based ranks."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC from scores (positive class = 1). None if one class."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return None
    ranks = average_ranks(np.asarray(scores, dtype=float))
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _discriminant(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Gaussian log-likelihood ratio for x_eval (positive = model).

    Params from training rows x, y. Diagonal covariance: the model class has
    ~84 rows, a 17x17 covariance would be noise.
    """
    mu1, mu0 = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    var1 = x[y == 1].var(axis=0) + VAR_FLOOR
    var0 = x[y == 0].var(axis=0) + VAR_FLOOR
    z1 = (x_eval - mu1) ** 2 / var1 + np.log(var1)
    z0 = (x_eval - mu0) ** 2 / var0 + np.log(var0)
    return -0.5 * (z1 - z0).sum(axis=1)


def _design(
    model_items: dict[str, dict[str, float]],
    rater_by_item: dict[str, list[dict[str, Any]]],
    dims: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """LOO-centered rows. X rows: human first, then model. Human rows are
    centered by the consensus of the *other* raters; the model row by the
    full consensus (the model is in no consensus).

    Returns (item_ids, X, y, row_item) where row_item maps every ROW of X
    to its item id - X has multiple rows per item, item_ids does not."""
    ids, rows, labels, row_item = [], [], [], []
    for item_id in sorted(model_items):
        rater_rows = [r for r in rater_by_item.get(item_id, ()) if all(d in r["ratings"] for d in dims)]
        pred = model_items[item_id]
        if len(rater_rows) < 2 or not all(d in pred for d in dims):
            continue
        full = np.array([float(np.mean([r["ratings"][d] for r in rater_rows])) for d in dims])
        for i, r in enumerate(rater_rows):
            others = [o for j, o in enumerate(rater_rows) if j != i]
            loo = np.array([float(np.mean([o["ratings"][d] for o in others])) for d in dims])
            rows.append([r["ratings"][d] - c for d, c in zip(dims, loo)])
            labels.append(0)
            row_item.append(item_id)
        rows.append([pred[d] - c for d, c in zip(dims, full)])
        labels.append(1)
        row_item.append(item_id)
        ids.append(item_id)
    return ids, np.array(rows, dtype=float), np.array(labels, dtype=int), np.array(row_item)


def mimicry(
    model_items: dict[str, dict[str, float]],
    rater_by_item: dict[str, list[dict[str, Any]]],
    dims: list[str] | None = None,
) -> dict[str, Any] | None:
    """Frozen human_mimicry for one model. None if not computable."""
    dims = dims or score.load_dimensions()
    ids, x, y, row_item = _design(model_items, rater_by_item, dims)
    if len(ids) < MIN_ITEMS:
        return None

    oof = np.zeros(len(x))
    for item_id in ids:
        held = np.where(row_item == item_id)[0]
        train = np.where(row_item != item_id)[0]
        oof[held] = _discriminant(x[train], y[train], x[held])
    auc = rank_auc(oof, y)
    if auc is None:
        return None
    sep = max(auc, 1 - auc)
    return {
        "human_mimicry": round(2 * (1 - sep), 4),
        "auc": round(auc, 4),
        "n_items": len(ids),
        "n_human_rows": int((y == 0).sum()),
        "n_model_rows": int((y == 1).sum()),
    }


def load_model_items(path: Path) -> dict[str, dict[str, float]]:
    """model_id -> per-item pred vectors, from a results file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    return {it["id"]: it["pred"] for it in d.get("per_item", []) if it.get("pred")}


def human_reference(
    rater_by_item: dict[str, list[dict[str, Any]]], dims: list[str] | None = None
) -> dict[str, float | None]:
    """Interpretive band: each real rater as the 'model' vs the other raters.

    Raters do NOT land at 1.0 - each has a fingerprint (per-dimension biases,
    scale usage) the discriminator learns from other items. The band defines
    what "as identifiable as an actual human" means; a model inside it is
    behaviorally exchangeable with an individual rater.
    """
    dims = dims or score.load_dimensions()
    out: dict[str, float | None] = {}
    raters = sorted({r["rater_id"] for rows in rater_by_item.values() for r in rows})
    for rater_id in raters:
        as_model = {
            iid: next(r["ratings"] for r in rows if r["rater_id"] == rater_id)
            for iid, rows in rater_by_item.items()
            if any(r["rater_id"] == rater_id for r in rows)
        }
        others = {
            iid: [r for r in rows if r["rater_id"] != rater_id]
            for iid, rows in rater_by_item.items()
        }
        res = mimicry(as_model, others, dims)
        out[rater_id] = res["human_mimicry"] if res else None
    return out


def human_band(
    rater_by_item: dict[str, list[dict[str, Any]]],
    item_ids: set[str],
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    """v3.0.0: the band published next to every human_mimicry score.

    Only raters who rated >= min_coverage of the holdout define the band; a
    rater with a handful of items gets an unstable score (rater_1: 16 of 84
    items, 0.43). Partial raters are still listed, marked excluded.
    """
    scoped = {k: v for k, v in rater_by_item.items() if k in item_ids}
    ref = human_reference(scoped)
    counts: dict[str, int] = {}
    for rows in scoped.values():
        for r in rows:
            counts[r["rater_id"]] = counts.get(r["rater_id"], 0) + 1
    full = [rid for rid, n in counts.items() if n >= min_coverage * len(scoped)]
    vals = [ref[rid] for rid in full if ref.get(rid) is not None]
    return {
        "band": [min(vals), max(vals)] if vals else None,
        "per_rater": ref,
        "in_band_raters": sorted(full),
        "coverage": counts,
    }


def main(argv: list[str]) -> int:
    rater_by_item = human_ceiling.load_rater_file()
    if "--reference" in argv:
        print("human reference (each rater as the 'model'):")
        for rater_id, m in human_reference(rater_by_item).items():
            print(f"  {rater_id:28} mimicry={'n/a' if m is None else f'{m:+.4f}'}")
        return 0
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        paths = [p for p in sorted((REPO / "results").glob("*.json"))
                 if p.name not in ("index.json", "human.json")]
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        out = mimicry(load_model_items(path), rater_by_item)
        if out:
            print(f"{d.get('model', path.stem)[:28]:30} mimicry={out['human_mimicry']:+.4f} "
                  f"auc={out['auc']:.4f} items={out['n_items']} "
                  f"human_rows={out['n_human_rows']}")
        else:
            print(f"{d.get('model', path.stem)[:28]:30} not_measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

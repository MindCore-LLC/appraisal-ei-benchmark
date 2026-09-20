"""Human ceiling from per-rater 17-dim vectors.

Leave-one-rater-out Pearson r against the mean of the other raters, averaged
over items. That is the number a model scored against consensus should be
compared to: a rater is never scored against a mean they helped create.

    python scoring/human_ceiling.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score

REPO = Path(__file__).resolve().parents[1]
RATER_FILE = REPO / "annotation" / "rater_vectors.jsonl"


def _mean_vector(vectors: list[dict[str, float]], dims: list[str]) -> dict[str, float]:
    out = {}
    for d in dims:
        vals = [v[d] for v in vectors if d in v]
        if vals:
            out[d] = float(np.mean(vals))
    return out


def leave_one_out(by_item: dict[str, list[dict[str, Any]]], dims: list[str] | None = None) -> dict[str, Any]:
    """LOO ceiling. `by_item` maps item id -> list of {rater_id, ratings}."""
    dims = dims or score.load_dimensions()
    per_item = []
    item_rs = []
    rater_rs: dict[str, list[float]] = defaultdict(list)
    pred_by_dim: dict[str, list[float]] = defaultdict(list)
    gold_by_dim: dict[str, list[float]] = defaultdict(list)

    for item_id, rows in sorted(by_item.items()):
        if len(rows) < 2:
            continue
        pair_rs = []
        for i, row in enumerate(rows):
            others = [rows[j]["ratings"] for j in range(len(rows)) if j != i]
            gold = _mean_vector(others, dims)
            r = score.ratings_correlation(row["ratings"], gold, dims)
            pair_rs.append(r)
            rater_rs[row.get("rater_id", str(i))].append(r)
            for d in dims:
                if d in row["ratings"] and d in gold:
                    pred_by_dim[d].append(float(row["ratings"][d]))
                    gold_by_dim[d].append(float(gold[d]))
        item_r = float(np.mean(pair_rs))
        item_rs.append(item_r)
        consensus = _mean_vector([row["ratings"] for row in rows], dims)
        per_item.append({
            "id": item_id,
            "r": round(item_r, 4),
            "n_raters": len(rows),
            "pred": {k: round(v, 4) for k, v in consensus.items()},
            "gold": {k: round(v, 4) for k, v in consensus.items()},
        })

    per_dim = []
    for d in dims:
        xs, ys = pred_by_dim[d], gold_by_dim[d]
        dim_r = None
        if len(xs) >= 8 and (max(xs) - min(xs) > 1e-8) and (max(ys) - min(ys) > 1e-8):
            dim_r = round(float(pearsonr(xs, ys)[0]), 4)
        per_dim.append({
            "id": d,
            "r": dim_r,
            "mae": round(float(np.mean(np.abs(np.array(xs) - np.array(ys)))), 3) if xs else None,
            "n": len(xs),
            "measured": dim_r is not None,
        })

    disc = score.discriminant_validity(dict(pred_by_dim), dict(gold_by_dim))
    cal = round(float(np.mean(item_rs)), 4) if item_rs else None
    return {
        "appraisal_calibration": cal,
        "discriminant_validity": round(disc, 4) if disc is not None else None,
        "n_items": len(per_item),
        "n_raters": len(rater_rs),
        "per_rater_mean_r": {k: round(float(np.mean(v)), 4) for k, v in sorted(rater_rs.items())},
        "per_dim": per_dim,
        "per_item": per_item,
        "pairwise_kappa": score.pairwise_kappa(
            {iid: [row["ratings"] for row in rows] for iid, rows in by_item.items() if len(rows) >= 2},
            dims,
        ),
    }


def load_rater_file(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    path = path or RATER_FILE
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_item[row["id"]].append(row)
    return dict(by_item)


def ceiling_for_ids(item_ids: set[str], path: Path | None = None) -> dict[str, Any]:
    by_item = {k: v for k, v in load_rater_file(path).items() if k in item_ids}
    return leave_one_out(by_item)


if __name__ == "__main__":
    holdout = REPO / "stimuli" / "text" / "vignettes_test_human.jsonl"
    ids = {json.loads(l)["id"] for l in holdout.read_text(encoding="utf-8").splitlines() if l.strip()}
    out = ceiling_for_ids(ids)
    print(json.dumps({k: out[k] for k in (
        "appraisal_calibration", "discriminant_validity", "n_items", "n_raters",
        "per_rater_mean_r", "pairwise_kappa")}, indent=2, default=str))

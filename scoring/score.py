"""Appraisal-EI Benchmark reference scorer (v1.1.0).

Standalone - numpy + scipy only, no benchmark-consumer dependencies.
Normative definitions live in SPEC.md section 3; this file is the executable
form of that section.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, wilcoxon

N_APPRAISAL = 17

AGGREGATE_KEYS = (
    "appraisal_calibration",
    "value_action",
    "persistence",
    "acoustic_risk_f1",
    "steering_score",
    "discriminant_validity",
    "human_mimicry",
)


def load_dimensions(schema_path: str | Path | None = None) -> list[str]:
    """Dimension ids in canonical order, from schema/rating_schema.json."""
    path = Path(schema_path) if schema_path else Path(__file__).resolve().parents[1] / "schema" / "rating_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    return [d["id"] for d in schema["dimensions"]]


def aggregate_ei(
    parts: dict[str, float | None], structural_zeros: tuple[str, ...] = ()
) -> float | None:
    """Unweighted mean over the sub-scores that were actually measured.

    A sub-score that is absent or None was NOT measured and is excluded from the
    mean - it is not evidence of a zero. A sub-score named in `structural_zeros`
    is a real zero by construction (e.g. `acoustic_risk_f1` for a text-only
    model, SPEC.md section 3) and is averaged in as 0.0.

    Returns None when nothing was measured. Always report the value alongside
    the count from `measured_subscores`, never bare: a mean over two sub-scores
    is not comparable to a mean over seven.

    Changed in 1.1.0: pre-1.1.0 this scored missing keys as 0.0, which silently
    diluted the aggregate by the number of unimplemented sub-scores.
    """
    vals = []
    for k in AGGREGATE_KEYS:
        if k in structural_zeros:
            vals.append(0.0)
            continue
        v = parts.get(k)
        if v is None:
            continue
        vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def measured_subscores(
    parts: dict[str, float | None], structural_zeros: tuple[str, ...] = ()
) -> dict[str, str]:
    """Per sub-score: 'measured', 'structural_zero', or 'not_measured'."""
    out = {}
    for k in AGGREGATE_KEYS:
        if k in structural_zeros:
            out[k] = "structural_zero"
        elif parts.get(k) is None:
            out[k] = "not_measured"
        else:
            out[k] = "measured"
    return out


def discriminant_validity(
    pred_by_dim: dict[str, list[float]],
    gold_by_dim: dict[str, list[float]],
    min_items: int = 8,
) -> float | None:
    """Convergent minus discriminant correlation across dimensions.

    Builds the multitrait matrix M[i][j] = r(pred[dim_i], gold[dim_j]) over
    items, then returns mean(diagonal) - mean(|off-diagonal|). A model that
    tracks each appraisal dimension specifically scores high; one that emits a
    single valence signal smeared across all 17 scores near zero, because its
    off-diagonal correlations are as strong as its diagonal ones.

    Returns None if fewer than 3 dimensions have >= min_items paired
    observations with variance on both sides.
    """
    usable = [
        d
        for d in pred_by_dim
        if d in gold_by_dim
        and len(pred_by_dim[d]) >= min_items
        and len(pred_by_dim[d]) == len(gold_by_dim[d])
        and max(pred_by_dim[d]) - min(pred_by_dim[d]) > 1e-8
        and max(gold_by_dim[d]) - min(gold_by_dim[d]) > 1e-8
    ]
    if len(usable) < 3:
        return None
    diag, off = [], []
    for i in usable:
        for j in usable:
            n = min(len(pred_by_dim[i]), len(gold_by_dim[j]))
            if n < min_items:
                continue
            r = float(pearsonr(pred_by_dim[i][:n], gold_by_dim[j][:n])[0])
            if math.isnan(r):
                continue
            (diag if i == j else off).append(r if i == j else abs(r))
    if not diag or not off:
        return None
    return float(np.mean(diag) - np.mean(off))


def parse_ratings(text: str, dimensions: list[str] | None = None) -> dict[str, float]:
    """Extract a rating vector from model output (first {...} JSON object)."""
    dims = dimensions or load_dimensions()
    if not text:
        return {}
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        raw = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}
    out = {}
    for d in dims:
        if d in raw:
            try:
                out[d] = float(raw[d])
            except (TypeError, ValueError):
                continue
    return out


def ratings_correlation(
    pred: dict[str, float], gold: dict[str, float], dimensions: list[str] | None = None
) -> float:
    """Pearson r between predicted and gold vectors over shared dims (<3 pairs -> 0)."""
    dims = dimensions or load_dimensions()
    xs, ys = [], []
    for d in dims:
        if d in pred and d in gold:
            xs.append(pred[d])
            ys.append(gold[d])
    if len(xs) < 3:
        return 0.0
    if max(xs) - min(xs) < 1e-8 or max(ys) - min(ys) < 1e-8:
        return 0.0
    return float(pearsonr(xs, ys)[0])


def _cohen_kappa(a: list[int], b: list[int]) -> float:
    """Unweighted kappa on integer bins. 0 if undefined."""
    n = len(a)
    if n == 0:
        return 0.0
    classes = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in classes)
    if abs(1 - pe) < 1e-9:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def pairwise_kappa(
    by_item: dict[str, list[dict[str, float]]], dimensions: list[str] | None = None
) -> float:
    """Mean kappa across dims, averaged over annotator pairs that co-rated an item."""
    dims = dimensions or load_dimensions()
    kappas = []
    for dim in dims:
        pairs: list[tuple[int, int]] = []
        for ratings in by_item.values():
            if len(ratings) < 2:
                continue
            bins = [int(round(r[dim])) for r in ratings]
            for i in range(len(bins)):
                for j in range(i + 1, len(bins)):
                    pairs.append((bins[i], bins[j]))
        if len(pairs) < 5:
            continue
        a, b = zip(*pairs)
        kappas.append(_cohen_kappa(list(a), list(b)))
    return float(sum(kappas) / len(kappas)) if kappas else float("nan")


def paired_test(eos_items: list[float], other_items: list[float]) -> dict[str, float]:
    """One-sided Wilcoxon signed-rank (eos > other). <8 pairs or identical -> p=1."""
    if len(eos_items) < 8 or np.allclose(eos_items, other_items):
        return {"p": 1.0, "d": 0.0}
    stat = wilcoxon(eos_items, other_items, alternative="greater")
    diff = np.array(eos_items) - np.array(other_items)
    d = float(diff.mean() / (diff.std() + 1e-8))
    return {"p": float(stat.pvalue), "d": d}


if __name__ == "__main__":
    print({"dimensions": len(load_dimensions()), "n_appraisal": N_APPRAISAL, "aggregate_keys": AGGREGATE_KEYS})

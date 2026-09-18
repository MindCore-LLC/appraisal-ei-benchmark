"""Appraisal-EI Benchmark reference scorer (v1.0.0).

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


def aggregate_ei(parts: dict[str, float]) -> float:
    """Unweighted mean of the seven sub-scores. Missing keys count as 0."""
    return float(np.mean([float(parts.get(k, 0.0)) for k in AGGREGATE_KEYS]))


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

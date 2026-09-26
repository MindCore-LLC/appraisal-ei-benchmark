"""Appraisal-EI Benchmark reference scorer (v3.0.0).

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

# Public leaderboard metrics. Everything else is a named roadmap slot, not a
# score that gets averaged into a headline.
# v3.0.0: headline is appraisal_tracking. appraisal_calibration (per-item r
# over the 17 dims) is kept as a diagnostic only: a constant average profile
# scores 0.864 on it, above every model and the human ceiling.
PUBLIC_KEYS = (
    "appraisal_tracking",
    "discriminant_validity",
)

DIAGNOSTIC_KEYS = ("appraisal_calibration",)

ROADMAP_KEYS = (
    "value_action",
    "persistence",
    "acoustic_risk_f1",
    "steering_score",
    "human_mimicry",
)

AGGREGATE_KEYS = PUBLIC_KEYS + DIAGNOSTIC_KEYS + ROADMAP_KEYS


def load_dimensions(schema_path: str | Path | None = None) -> list[str]:
    """Dimension ids in canonical order, from schema/rating_schema.json."""
    path = Path(schema_path) if schema_path else Path(__file__).resolve().parents[1] / "schema" / "rating_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    return [d["id"] for d in schema["dimensions"]]


def aggregate_ei(
    parts: dict[str, float | None], structural_zeros: tuple[str, ...] = ()
) -> float | None:
    """Mean of the *public* metrics that were actually measured.

    Headline rank is `appraisal_tracking`, not this value. This exists so a
    two-number summary (tracking + discriminant validity) can be quoted
    together. Diagnostic keys (appraisal_calibration) are never averaged in. Roadmap keys and structural zeros are never averaged in: a
    text-only run is not an acoustic failure, and an unimplemented sub-score
    is not a zero.

    `structural_zeros` is accepted for back-compat with v1 callers and is
    ignored. Status still records them via `measured_subscores`.

    Changed in 3.0.0: public metrics are tracking + discriminant validity.
    Changed in 2.0.0: structural zeros no longer dilute the mean.
    Changed in 1.1.0: missing keys stopped scoring as 0.0.
    """
    del structural_zeros  # recorded in status, never averaged
    vals = []
    for k in PUBLIC_KEYS:
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


def appraisal_tracking(
    pred_by_dim: dict[str, list[float]],
    gold_by_dim: dict[str, list[float]],
    min_items: int = 8,
) -> float | None:
    """Headline (v3.0.0): mean over dims of Pearson r across items.

    Asks "when the situation changes, does the model's rating move with the
    humans'?" per dimension. A dim the gold does not vary on, or with fewer
    than min_items pairs, is skipped (not measurable). A dim the MODEL does not
    vary on scores 0.0, not skipped: a constant carries no information, so an
    average-profile answer scores 0 instead of topping the board.

    None if fewer than 3 dims are measurable.
    """
    rs = []
    for d, ys in gold_by_dim.items():
        xs = pred_by_dim.get(d) or []
        if len(xs) < min_items or len(xs) != len(ys) or max(ys) - min(ys) < 1e-8:
            continue
        rs.append(0.0 if max(xs) - min(xs) < 1e-8 else float(pearsonr(xs, ys)[0]))
    return float(np.mean(rs)) if len(rs) >= 3 else None


def scenario_map(unique_path: str | Path | None = None) -> dict[str, str]:
    """item id -> scenario id. The 84 holdout rows are 23 name-swapped
    scenarios, so resampling must be by scenario, not by row."""
    path = Path(unique_path) if unique_path else Path(__file__).resolve().parents[1] / "stimuli" / "text" / "vignettes_unique_human.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            u = json.loads(line)
            for m in u["collapsed_from"]:
                out[m] = u["id"]
    return out


def tracking_ci(
    items: list[dict], dims: list[str], n_boot: int = 2000, seed: int = 0
) -> list[float] | None:
    """95% CI for appraisal_tracking, resampling whole scenarios."""
    smap = scenario_map()
    usable = [it for it in items if it.get("pred") and it.get("gold")]
    groups: dict[str, list[dict]] = {}
    for it in usable:
        groups.setdefault(smap.get(it["id"], it["id"]), []).append(it)
    keys = sorted(groups)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = [it for k in rng.choice(keys, len(keys)) for it in groups[k]]
        p = {d: [float(it["pred"][d]) for it in pick if d in it["pred"] and d in it["gold"]] for d in dims}
        g = {d: [float(it["gold"][d]) for it in pick if d in it["pred"] and d in it["gold"]] for d in dims}
        v = appraisal_tracking(p, g)
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


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

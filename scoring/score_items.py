"""One place that turns stored per-item pred/gold vectors into every score a
results file carries (v3.0.0). Used by run_eval.py (new runs) and
scripts/backfill_results.py (re-scoring stored runs), so the two never drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scipy.stats import pearsonr

import human_mimicry
import score

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "stimuli" / "text" / "vignettes_test_human.jsonl"


def holdout_ids() -> set[str]:
    return {json.loads(l)["id"] for l in HOLDOUT.read_text(encoding="utf-8").splitlines() if l.strip()}


def score_items(items: list[dict[str, Any]], dims: list[str] | None = None) -> dict[str, Any]:
    dims = dims or score.load_dimensions()
    pred_by, gold_by, per_dim = {}, {}, []
    for dim in dims:
        xs, ys, errs = [], [], []
        for it in items:
            p, g = it.get("pred"), it.get("gold")
            if p and g and dim in p and dim in g:
                xs.append(float(p[dim]))
                ys.append(float(g[dim]))
                errs.append(abs(float(p[dim]) - float(g[dim])))
        pred_by[dim], gold_by[dim] = xs, ys
        r = None
        if len(xs) >= 8 and max(xs) - min(xs) > 1e-8 and max(ys) - min(ys) > 1e-8:
            r = round(float(pearsonr(xs, ys)[0]), 4)
        per_dim.append({"id": dim, "r": r, "mae": round(sum(errs) / len(errs), 3) if errs else None,
                        "n": len(xs), "measured": r is not None})

    corrs = [it["r"] for it in items if isinstance(it.get("r"), (int, float))]
    cal = round(sum(corrs) / len(corrs), 4) if corrs else None
    trk = score.appraisal_tracking(pred_by, gold_by)
    disc = score.discriminant_validity(pred_by, gold_by)
    raters = human_mimicry.human_ceiling.load_rater_file()
    mim = human_mimicry.mimicry({it["id"]: it["pred"] for it in items if it.get("pred")}, raters)
    band = human_mimicry.human_band(raters, holdout_ids())["band"]

    sub = {
        "appraisal_tracking": round(trk, 4) if trk is not None else None,
        "discriminant_validity": round(disc, 4) if disc is not None else None,
        "appraisal_calibration": cal,
        "value_action": None, "persistence": None, "acoustic_risk_f1": None, "steering_score": None,
        "human_mimicry": mim["human_mimicry"] if mim else None,
    }
    st = score.measured_subscores(sub)
    agg = score.aggregate_ei(sub)
    return {
        "appraisal_tracking": sub["appraisal_tracking"],
        "appraisal_tracking_ci95": score.tracking_ci(items, dims) if trk is not None else None,
        "appraisal_calibration": cal,
        "human_mimicry_band": band,
        "per_dim": per_dim,
        "subscores": sub,
        "subscore_status": st,
        "n_subscores_measured": sum(1 for k in score.PUBLIC_KEYS if st.get(k) == "measured"),
        "n_subscores_total": len(score.PUBLIC_KEYS),
        "aggregate_ei": round(agg, 4) if agg is not None else None,
    }

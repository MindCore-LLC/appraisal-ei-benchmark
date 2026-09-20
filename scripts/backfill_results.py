#!/usr/bin/env python3
"""Recompute per_dim / sub-scores in results/*.json under the 1.1.0 rules.

Uses the stored per_item pred+gold vectors, so no model is re-run. Turns
not-measured dimensions from 0.0 into null and adds discriminant_validity,
subscore_status and aggregate_ei.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoring"))
import human_mimicry
import score
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
dims = score.load_dimensions()
VERSION = (REPO / "VERSION").read_text(encoding="utf-8").strip()


def redo(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    items = d.get("per_item")
    if not items:
        return f"{path.name}: skipped (no per_item)"

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
    disc = score.discriminant_validity(pred_by, gold_by)
    mim = human_mimicry.mimicry(
        {it["id"]: it["pred"] for it in items if it.get("pred")},
        human_mimicry.human_ceiling.load_rater_file(),
    )
    sub = {"appraisal_calibration": cal, "value_action": None, "persistence": None,
           "acoustic_risk_f1": None, "steering_score": None,
           "discriminant_validity": round(disc, 4) if disc is not None else None,
           "human_mimicry": mim["human_mimicry"] if mim else None}
    st = score.measured_subscores(sub)
    agg = score.aggregate_ei(sub)

    was_zero = sum(1 for x in d.get("per_dim", []) if x.get("r") == 0.0)
    now_null = sum(1 for x in per_dim if x["r"] is None)
    d.update({"benchmark_version": VERSION, "appraisal_calibration": cal, "per_dim": per_dim,
              "subscores": sub, "subscore_status": st,
              "n_subscores_measured": sum(1 for k in score.PUBLIC_KEYS if st.get(k) == "measured"),
              "n_subscores_total": len(score.PUBLIC_KEYS),
              "aggregate_ei": round(agg, 4) if agg is not None else None})
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return (f"{d['model'][:22]:24} cal={cal} disc={sub['discriminant_validity']} "
            f"agg={d['aggregate_ei']} ({d['n_subscores_measured']}/{d['n_subscores_total']})  "
            f"per_dim 0.0->null: {was_zero}->{now_null}")


def main() -> int:
    out = REPO / "results"
    for f in sorted(out.glob("*.json")):
        if f.name in ("index.json", "human.json"):
            continue
        print(redo(f))
    files = sorted(f.name for f in out.glob("*.json") if f.name != "index.json")
    (out / "index.json").write_text(json.dumps({"results": files}, indent=2), encoding="utf-8")
    print(f"rewrote results at v{VERSION}; run python site/build_site.py for the site payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

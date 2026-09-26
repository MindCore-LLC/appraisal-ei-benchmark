#!/usr/bin/env python3
"""Re-score every results/*.json under the current VERSION rules.

Uses the stored per_item pred+gold vectors, so no model is re-run. All scores
come from scoring/score_items.py, the same code run_eval.py uses. The human
row is rebuilt from the rater vectors (human_ceiling.py).

    python scripts/backfill_results.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoring"))
import human_ceiling
import human_mimicry
import score_items

REPO = Path(__file__).resolve().parents[1]
VERSION = (REPO / "VERSION").read_text(encoding="utf-8").strip()


def redo(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    items = d.get("per_item")
    if not items:
        return f"{path.name}: skipped (no per_item)"
    d.update({"benchmark_version": VERSION, **score_items.score_items(items)})
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    s = d["subscores"]
    return (f"{d['model'][:26]:28} track={s['appraisal_tracking']} ci={d['appraisal_tracking_ci95']} "
            f"disc={s['discriminant_validity']} cal(diag)={s['appraisal_calibration']} mim={s['human_mimicry']}")


def redo_human(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    ceil = human_ceiling.ceiling_for_ids(score_items.holdout_ids())
    band = human_mimicry.human_band(human_ceiling.load_rater_file(), score_items.holdout_ids())
    d["subscores"].update({"appraisal_tracking": ceil["appraisal_tracking"]})
    d.update({
        "benchmark_version": VERSION,
        "appraisal_tracking": ceil["appraisal_tracking"],
        "per_rater_tracking": ceil["per_rater_tracking"],
        "human_mimicry_band": band["band"],
        "human_mimicry_per_rater": band["per_rater"],
        "note": ("Leave-one-rater-out: each rater scored against the mean of the other raters, "
                 "exactly as a model is. Tracking ceiling = mean over raters covering >= 80% of the "
                 "84-item holdout. Not a model."),
    })
    d["subscore_status"] = {k: ("measured" if d["subscores"].get(k) is not None else "not_measured") for k in d["subscores"]}
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return f"{'Human ceiling':28} track={d['appraisal_tracking']} per_rater={d['per_rater_tracking']} mimicry band={band['band']}"


def main() -> int:
    out = REPO / "results"
    for f in sorted(out.glob("*.json")):
        if f.name == "index.json":
            continue
        print(redo_human(f) if f.name == "human.json" else redo(f))
    files = sorted(f.name for f in out.glob("*.json") if f.name != "index.json")
    (out / "index.json").write_text(json.dumps({"results": files}, indent=2), encoding="utf-8")
    print(f"rewrote results at v{VERSION}; run python site/build_site.py for the site payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

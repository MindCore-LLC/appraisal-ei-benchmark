#!/usr/bin/env python3
"""Pack round-3 human ratings into the public benchmark artifacts.

Reads mindcore-poc appraisal_labeled exports (not committed here), writes:

- annotation/rater_vectors.jsonl     per-rater 17-dim vectors, anonymized ids
- stimuli/text/vignettes_unique_human.jsonl
                                     one row per distinct scenario (names masked)
- results/human.json                 leave-one-out ceiling on the frozen 84-item holdout
- HUMAN_CEILING.md                   the number, the method, how to recompute

Emails never leave this process.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scoring"))
import human_ceiling
import score

DEFAULT_SRC = REPO.parent / "mindcore-poc" / "data" / "appraisal_labeled"

NAMES = (r"\b(Casey|Morgan|Avery|Sam|Riley|Jordan|Alex|Devon|Emerson|Quinn|Drew|Finley"
         r"|Taylor|Reese|Rowan|Skyler|Harper|Blake|Cameron|Parker|Sage|Hayden|Elliot"
         r"|Noel|Kai|Remy)\b")
norm = lambda t: re.sub(r"\s+", " ", re.sub(NAMES, "X", t)).strip().lower()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def rater_map(emails: list[str]) -> dict[str, str]:
    unique = sorted({e.strip().lower() for e in emails if e})
    return {e: f"rater_{i + 1}" for i, e in enumerate(unique)}


def pack(src: Path) -> dict:
    dims = score.load_dimensions()
    consensus = _load_jsonl(src / "human_ratings.jsonl")
    raw = _load_jsonl(src / "annotations_round3_raw.jsonl")
    holdout_path = REPO / "stimuli" / "text" / "vignettes_test_human.jsonl"
    holdout = _load_jsonl(holdout_path)
    holdout_ids = {r["id"] for r in holdout}

    emails = [row["annotator"] for row in raw if row.get("annotator")]
    amap = rater_map(emails)

    # Per-rater vectors at item level (the texts models actually saw).
    rater_rows = []
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in raw:
        vid = row.get("vignette_id") or row.get("id")
        ratings = row.get("ratings")
        if not vid or not ratings:
            continue
        rid = amap[row["annotator"].strip().lower()]
        rec = {"id": vid, "rater_id": rid, "split": row.get("split"), "ratings": ratings}
        rater_rows.append(rec)
        by_item[vid].append(rec)
    rater_rows.sort(key=lambda r: (r["id"], r["rater_id"]))
    _write_jsonl(REPO / "annotation" / "rater_vectors.jsonl", rater_rows)

    # Unique-scenario gold: name-masked text, mean of item-level consensus.
    text_by_id = {r["id"]: r["text"] for r in consensus}
    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in consensus:
        clusters[norm(r["text"])].append(r)

    unique = []
    for members in clusters.values():
        members.sort(key=lambda r: r["id"])
        canon = members[0]
        ratings = {}
        for d in dims:
            vals = [m["ratings"][d] for m in members if d in (m.get("ratings") or {})]
            if vals:
                ratings[d] = round(statistics.mean(vals), 4)
        unique.append({
            "id": canon["id"],
            "text": canon["text"],
            "ratings": ratings,
            "source": "human_core_consensus",
            "annotators": int(round(statistics.mean(m.get("annotators", 0) for m in members))) if members else None,
            "n_source_items": len(members),
            "collapsed_from": [m["id"] for m in members],
            "in_holdout": any(m["id"] in holdout_ids for m in members),
        })
    unique.sort(key=lambda r: r["id"])
    _write_jsonl(REPO / "stimuli" / "text" / "vignettes_unique_human.jsonl", unique)

    # Ceiling on the same 84 holdout items the models were scored on.
    ceiling = human_ceiling.leave_one_out(
        {k: v for k, v in by_item.items() if k in holdout_ids}, dims
    )
    cal = ceiling["appraisal_calibration"]
    disc = ceiling["discriminant_validity"]
    sub = {
        "appraisal_calibration": cal,
        "value_action": None,
        "persistence": None,
        "acoustic_risk_f1": None,
        "steering_score": None,
        "discriminant_validity": disc,
        "human_mimicry": None,
    }
    st = score.measured_subscores(sub)  # no structural zeros: humans are not a text-only model
    agg = score.aggregate_ei(sub)
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    human_result = {
        "model": "Human ceiling",
        "model_id": "human",
        "provider": "none",
        "benchmark_version": version,
        "stimuli": "stimuli/text/vignettes_test_human.jsonl",
        "gold_source": "human_core_consensus",
        "status": "measured",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_items": ceiling["n_items"],
        "n_parsed": ceiling["n_items"],
        "appraisal_calibration": cal,
        "note": (
            "Leave-one-rater-out Pearson r vs the mean of the other raters, "
            "averaged over the frozen 84-item holdout. Not a model."
        ),
        "n_raters": ceiling["n_raters"],
        "per_rater_mean_r": ceiling["per_rater_mean_r"],
        "pairwise_kappa": None if (isinstance(ceiling["pairwise_kappa"], float) and ceiling["pairwise_kappa"] != ceiling["pairwise_kappa"]) else (
            round(float(ceiling["pairwise_kappa"]), 4) if ceiling["pairwise_kappa"] is not None else None
        ),
        "subscores": sub,
        "subscore_status": st,
        "n_subscores_measured": sum(1 for v in st.values() if v == "measured"),
        "n_subscores_total": len(score.PUBLIC_KEYS),
        "aggregate_ei": round(agg, 4) if agg is not None else None,
        "per_dim": ceiling["per_dim"],
        "per_item": ceiling["per_item"],
    }
    (REPO / "results" / "human.json").write_text(json.dumps(human_result, indent=2), encoding="utf-8")

    md = f"""# Human ceiling

Leave-one-rater-out Pearson r against the mean of the other raters, on the
frozen holdout `stimuli/text/vignettes_test_human.jsonl`.

| | |
|---|---|
| **Calibration (headline)** | **{cal}** |
| Discriminant validity | {disc} |
| Items with ≥2 raters | {ceiling["n_items"]} |
| Raters | {ceiling["n_raters"]} |
| Pairwise unweighted κ | {human_result["pairwise_kappa"]} |

Per-rater mean LOO r: `{json.dumps(ceiling["per_rater_mean_r"])}`

## Why leave-one-out

A model is scored against the consensus of the raters. Scoring a rater against
a consensus they helped create inflates the ceiling. LOO is the comparable
number: each rater is treated the way a model is treated.

Do **not** claim a model "beats humans" if its calibration vs full consensus sits
next to this LOO number. Those are neighbouring measurements, not the same
test. The honest sentence is: frontier models are in the same band as a
held-out rater.

Unweighted pairwise κ on a 7-point scale is expected to look low; adjacent-point
disagreement is common. The scoring metric is Pearson r, so the ceiling is r.

## Recompute

```
python scoring/human_ceiling.py
python scripts/pack_human_gold.py
```

Source vectors: `annotation/rater_vectors.jsonl` (anonymized `rater_N` ids).
Round-3 emails never enter this repo.

## Unique scenarios

`stimuli/text/vignettes_unique_human.jsonl` collapses name-swapped duplicates
to **{len(unique)}** distinct texts (from {len(consensus)} rated rows). The
84-item holdout covers **{sum(1 for r in unique if r["in_holdout"])}** of those
unique scenarios. Public rank still uses the 84 rows so published model scores
stay comparable. Do not quote the 560-row file as 560 scenarios.
"""
    (REPO / "HUMAN_CEILING.md").write_text(md, encoding="utf-8")

    return {
        "n_consensus": len(consensus),
        "n_unique": len(unique),
        "n_holdout": len(holdout_ids),
        "n_rater_rows": len(rater_rows),
        "n_raters": len(amap),
        "calibration": cal,
        "discriminant_validity": disc,
        "holdout_in_unique": sum(1 for r in unique if r["in_holdout"]),
    }


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    summary = pack(src)
    print(json.dumps(summary, indent=2))

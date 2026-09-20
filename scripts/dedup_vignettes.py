#!/usr/bin/env python3
"""Collapse vignettes_*.jsonl (560 rows, 94 real scenarios) into a diagnostic set.

560 rows are 94 distinct texts x 2 names x 3 intensity levels, and the text is
byte-identical across intensity levels (see stimuli/text/QC_FINDINGS.md). This
emits one row per distinct text.

Priors that differed only by intensity are averaged, because they described the
same sentence; `prior_spread` records how far apart they were, so the
incoherence stays visible rather than being laundered by the mean.

Writes stimuli/text/vignettes_diagnostic.jsonl. No split: nothing is trained
here, so carving a holdout only discards statistical power.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STIM = REPO / "stimuli" / "text"
sys.path.insert(0, str(REPO / "scoring"))
import score

NAMES = (r"\b(Casey|Morgan|Avery|Sam|Riley|Jordan|Alex|Devon|Emerson|Quinn|Drew|Finley"
         r"|Taylor|Reese|Rowan|Skyler|Harper|Blake|Cameron|Parker|Sage|Hayden|Elliot"
         r"|Noel|Kai|Remy)\b")
norm = lambda t: re.sub(r"\s+", " ", re.sub(NAMES, "X", t)).strip().lower()


def main() -> int:
    dims = score.load_dimensions()
    rows = []
    for p in sorted(STIM.glob("vignettes_*.jsonl")):
        if p.name == "vignettes_diagnostic.jsonl":
            continue
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    clusters = defaultdict(list)
    for r in rows:
        clusters[norm(r["text"])].append(r)

    out = []
    for members in clusters.values():
        members.sort(key=lambda r: r["id"])
        canon = members[0]
        ratings, spread = {}, {}
        for d in dims:
            vals = [m["ratings"][d] for m in members if d in (m.get("ratings") or {})]
            if not vals:
                continue
            ratings[d] = round(statistics.mean(vals), 4)
            spread[d] = round(max(vals) - min(vals), 4)
        out.append({
            "id": canon["id"],
            "family": canon["family"],
            "text": canon["text"],
            "ratings": ratings,
            "prior_spread": spread,
            "ratings_provenance": "generator_priors",
            "split": "diagnostic",
            "collapsed_from": [m["id"] for m in members],
            "intensity_levels_collapsed": sorted({m["intensity"] for m in members}),
        })
    out.sort(key=lambda r: r["id"])

    dest = STIM / "vignettes_diagnostic.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")

    const = [d for d in dims if len({r["ratings"].get(d) for r in out}) == 1]
    worst = sorted(((max(r["prior_spread"].values(), default=0), r["id"]) for r in out), reverse=True)[:3]
    print(f"wrote {dest.relative_to(REPO)}: {len(out)} scenarios from {len(rows)} rows")
    print(f"  families: {len({r['family'] for r in out})}")
    print(f"  zero-variance dimensions remaining: {const or 'none'}")
    print(f"  largest prior disagreement within a collapsed cluster: {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

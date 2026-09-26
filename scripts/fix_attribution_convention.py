#!/usr/bin/env python3
"""One-time correction (v3.0.0): put human attribution gold on the schema's scale.

The round-3 rating form anchored attribution as -3 "their own character/choices"
... +3 "other people's actions"; rating_schema.json anchors it the other way
(+3 "caused by the person themselves"). Evidence the gold followed the form:
r(responsibility, attribution) = -0.80 across rater vectors, and frontier
models score negative r on attribution. This script negates attribution in
every file that stores human vignette gold, and marks each row with
"attribution_convention": "schema" so it can never be flipped twice.
Circumstance, which the form put at 0, stays at 0 (the schema puts it at -3);
that residual is documented in SPEC.md.

    python scripts/fix_attribution_convention.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARK = "attribution_convention"


def flip_rows(path: Path, get) -> tuple[int, int]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = skipped = 0
    for r in rows:
        if r.get(MARK) == "schema":
            skipped += 1
            continue
        for ratings in get(r):
            if ratings and "attribution" in ratings and ratings["attribution"] is not None:
                ratings["attribution"] = -float(ratings["attribution"])
        r[MARK] = "schema"
        done += 1
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return done, skipped


def main() -> None:
    text = REPO / "stimuli" / "text"
    for f in ("vignettes_test_human.jsonl", "vignettes_unique_human.jsonl"):
        print(f, flip_rows(text / f, lambda r: [r.get("ratings")]))
    print("rater_vectors.jsonl", flip_rows(REPO / "annotation" / "rater_vectors.jsonl", lambda r: [r.get("ratings")]))
    # stored model results: gold lives in per_item; predictions are untouched
    for f in sorted((REPO / "results").glob("*.json")):
        if f.name == "index.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if not str(d.get("stimuli", "")).endswith("vignettes_test_human.jsonl") or d.get(MARK) == "schema":
            continue
        for it in d.get("per_item") or []:
            g = it.get("gold")
            if g and "attribution" in g:
                g["attribution"] = -float(g["attribution"])
            if d.get("model_id") == "human" and it.get("pred") and "attribution" in it["pred"]:
                it["pred"]["attribution"] = -float(it["pred"]["attribution"])  # human row's pred is the consensus
        d[MARK] = "schema"
        f.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print("results/" + f.name, "gold flipped")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the two no-model baseline rows onto the board (v3.0.0).

    baseline-average-profile.json   the same 17-dim vector for every item: the
                                    mean train-split human consensus. Scores
                                    0 on appraisal_tracking by construction and
                                    0.864 on the old calibration headline -
                                    that gap is why v3 changed the headline.
    baseline-average-noise.json     the average profile plus Gaussian noise
                                    with the per-dim spread of real raters
                                    around their LOO consensus (train split),
                                    rounded to the integer scale. Anything that
                                    scores like this on human_mimicry is not
                                    evidence of human-likeness.

Both use train-split raters only; nothing is fit on the holdout. Seeded, so
the rows are reproducible. Run backfill_results.py afterwards to score them.

    python scripts/make_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoring"))
import human_ceiling
import score

REPO = Path(__file__).resolve().parents[1]
SEED = 0


def main() -> int:
    dims = score.load_dimensions()
    raters = human_ceiling.load_rater_file()
    train = {k: v for k, v in raters.items() if v and v[0].get("split") == "train" and len(v) >= 2}

    consensus = [[np.mean([r["ratings"][d] for r in rows]) for d in dims] for rows in train.values()]
    mu = np.mean(consensus, axis=0)
    dev = []
    for rows in train.values():
        for i, r in enumerate(rows):
            others = [o for j, o in enumerate(rows) if j != i]
            dev.append([r["ratings"][d] - np.mean([o["ratings"][d] for o in others]) for d in dims])
    sd = np.std(dev, axis=0)

    holdout = [json.loads(l) for l in (REPO / "stimuli" / "text" / "vignettes_test_human.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(SEED)
    rows = {
        "average-profile": ("Baseline: average profile", [mu for _ in holdout]),
        "average-noise": ("Baseline: average + noise", [np.clip(np.round(mu + rng.normal(0, 1, len(dims)) * sd), -3, 3) for _ in holdout]),
    }
    for slug, (name, preds) in rows.items():
        items = []
        for h, p in zip(holdout, preds):
            pred = {d: round(float(v), 4) for d, v in zip(dims, p)}
            items.append({"id": h["id"], "r": score.ratings_correlation(pred, h["ratings"], dims), "pred": pred, "gold": h["ratings"]})
        out = {
            "model": name, "model_id": f"baseline/{slug}", "provider": "baseline",
            "stimuli": "stimuli/text/vignettes_test_human.jsonl", "gold_source": "human_core_consensus",
            "status": "baseline", "n_items": len(items), "n_parsed": len(items),
            "note": (__doc__ or "").split("\n\n")[1].strip(), "seed": SEED, "per_item": items,
        }
        (REPO / "results" / f"baseline-{slug}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote", f"results/baseline-{slug}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

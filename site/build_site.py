"""Consolidate results/*.json + models.json -> docs/data.json for the leaderboard site.

GitHub Pages serves the docs/ tree only, so the page can't reach ../results/.
This merges everything into one static payload the page fetches once.

    python site/build_site.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scoring"))


def _direction_pct(result: dict) -> float | None:
    """% of (item, dim) ratings where pred sign matches gold sign.

    A true percentage a lay reader understands - "how often did the model
    point the same direction?" Pairs where either rating is exactly 0 are
    skipped (no direction to compare).
    """
    match = total = 0
    for it in result.get("per_item") or []:
        pred, gold = it.get("pred"), it.get("gold")
        if not pred or not gold:
            continue
        for d, g in gold.items():
            if d not in pred:
                continue
            p = float(pred[d])
            g = float(g)
            if p == 0.0 or g == 0.0:
                continue
            total += 1
            if (p > 0) == (g > 0):
                match += 1
    return round(100 * match / total, 1) if total else None


def _cost_per_item(result: dict, meta: dict) -> float | None:
    """Mean USD per parsed item from the run's token usage and registry pricing.

    Needs all four inputs (in/out price, prompt/completion tokens); anything
    missing -> None, and the cost axis hides that model rather than guessing.
    """
    p = meta.get("pricing") or {}
    i_in, i_out = p.get("input_per_1m"), p.get("output_per_1m")
    rt = result.get("runtime") or {}
    pt, ct = rt.get("prompt_tokens"), rt.get("completion_tokens")
    n = result.get("n_parsed")
    if None in (i_in, i_out, pt, ct) or not n:
        return None
    return round((pt * i_in + ct * i_out) / 1e6 / n, 6)


def main() -> None:
    registry = json.loads((REPO / "models.json").read_text(encoding="utf-8"))["models"]
    results = []
    for f in sorted((REPO / "results").glob("*.json")):
        if f.name == "index.json":
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        meta = registry.get(r.get("model_id"), {})
        r["meta"] = meta
        r["direction_pct"] = _direction_pct(r)
        r["cost_per_item_usd"] = _cost_per_item(r, meta)
        results.append(r)

    # Smoke stays in the payload so the "show provisional" toggle works; the
    # site hides it by default. Archive lives under results/archive/ and is
    # never globbed.
    #
    # The human reference band for human_mimicry is computed here (not
    # hardcoded in the page) so the board and the note always agree with the
    # rater vectors.
    import human_mimicry

    ref = human_mimicry.human_reference(human_mimicry.human_ceiling.load_rater_file())
    band = [v for v in ref.values() if v is not None]
    payload = {
        "benchmark_version": (REPO / "VERSION").read_text().strip(),
        "headline": "appraisal_calibration",
        "public_metrics": ["appraisal_calibration", "discriminant_validity"],
        "human_reference": {
            "mimicry": ref,
            "mimicry_band": [min(band), max(band)] if band else None,
        },
        "results": results,
    }
    out = REPO / "docs" / "data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"models": len(results), "wrote": str(out)})


if __name__ == "__main__":
    main()

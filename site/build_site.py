"""Consolidate results/*.json + models.json -> docs/data.json for the leaderboard site.

GitHub Pages serves the docs/ tree only, so the page can't reach ../results/.
This merges everything into one static payload the page fetches once.

    python site/build_site.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


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
        results.append(r)

    payload = {
        "benchmark_version": (REPO / "VERSION").read_text().strip(),
        "results": results,
    }
    out = REPO / "docs" / "data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"models": len(results), "wrote": str(out)})


if __name__ == "__main__":
    main()

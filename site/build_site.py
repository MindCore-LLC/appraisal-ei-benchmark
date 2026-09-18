"""Consolidate results/*.json + models.json -> docs/data.json for the leaderboard site.

GitHub Pages serves the docs/ tree only, so the page can't reach ../results/.
This merges everything into one static payload the page fetches once.

    python site/build_site.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    registry = json.loads((REPO / "models.json").read_text(encoding="utf-8"))["models"]
    results = []
    for f in sorted((REPO / "results").glob("*.json")):
        if f.name == "index.json":
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        meta = registry.get(r.get("model_id"), {})
        r["meta"] = meta
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

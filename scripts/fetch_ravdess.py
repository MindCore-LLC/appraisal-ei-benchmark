"""Map manifest wav_refs onto a local RAVDESS download.

RAVDESS is research-licensed and not redistributed with this benchmark.
Point this at your copy (the standard Zenodo release, `Audio_Speech_Actors_01-24`)
and it verifies coverage of stimuli/audio/manifest_v1.jsonl.

Usage:
    python scripts/fetch_ravdess.py --root /path/to/ravdess
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="RAVDESS root containing Actor_01..Actor_24")
    args = p.parse_args()
    root = Path(args.root)
    manifest = Path(__file__).resolve().parents[1] / "stimuli" / "audio" / "manifest_v1.jsonl"

    missing = []
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        wav = root / r["wav_ref"]
        if not wav.exists():
            missing.append(r["wav_ref"])
    print({"clips": len(rows), "missing": len(missing), "root": str(root)})
    if missing:
        for m in missing[:10]:
            print("  missing:", m)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""v2.0.0+ artifact checks (updated for v3.0.0). Run: python scoring/test_v2_gold.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_version_aligned():
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    assert version == "3.0.0", version
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "Specification v3.0.0" in spec
    gs = json.loads((REPO / "stimuli" / "text" / "gold_status.json").read_text(encoding="utf-8"))
    assert gs["version"] == "3.0.0"


def test_unique_scenario_gold():
    unique = _jsonl(REPO / "stimuli" / "text" / "vignettes_unique_human.jsonl")
    holdout = _jsonl(REPO / "stimuli" / "text" / "vignettes_test_human.jsonl")
    assert len(unique) == 94, len(unique)
    assert len(holdout) == 84, len(holdout)
    assert all(r.get("source") == "human_core_consensus" for r in unique)
    assert sum(1 for r in unique if r.get("in_holdout")) == 23


def test_rater_vectors_anonymized():
    rows = _jsonl(REPO / "annotation" / "rater_vectors.jsonl")
    assert len(rows) == 1772, len(rows)
    blob = (REPO / "annotation" / "rater_vectors.jsonl").read_text(encoding="utf-8")
    assert "@" not in blob
    raters = {r["rater_id"] for r in rows}
    assert raters == {"rater_1", "rater_2", "rater_3", "rater_4"}


def test_human_ceiling_row():
    human = json.loads((REPO / "results" / "human.json").read_text(encoding="utf-8"))
    assert human["model_id"] == "human"
    assert human["status"] == "measured"
    cal = human["appraisal_calibration"]
    assert 0.7 < cal < 0.95, cal
    assert human["n_items"] == 84
    assert human["n_subscores_total"] == 2
    assert 0.5 < human["appraisal_tracking"] < 0.75, human["appraisal_tracking"]
    lo, hi = human["human_mimicry_band"]
    assert 0 <= lo < hi <= 1


def test_attribution_on_schema_scale():
    """Human attribution gold must run with responsibility (schema: +3 = self)."""
    import numpy as np
    rows = _jsonl(REPO / "annotation" / "rater_vectors.jsonl")
    assert all(r.get("attribution_convention") == "schema" for r in rows)
    x = np.array([r["ratings"]["responsibility"] for r in rows])
    y = np.array([r["ratings"]["attribution"] for r in rows])
    assert np.corrcoef(x, y)[0, 1] > 0.5


def test_v3_baselines_expose_old_headline():
    avg = json.loads((REPO / "results" / "baseline-average-profile.json").read_text(encoding="utf-8"))
    human = json.loads((REPO / "results" / "human.json").read_text(encoding="utf-8"))
    # the reason for v3: a constant profile beat the human row on calibration
    assert avg["appraisal_calibration"] > human["appraisal_calibration"]
    assert avg["appraisal_tracking"] == 0.0
    assert avg["status"] == "baseline"


def test_smoke_archived():
    assert (REPO / "results" / "archive" / "meta-llama-llama-3-3-70b-instruct-turbo.json").exists()
    # The generator-prior smoke lives in archive/. The issue-1 re-run recreated
    # this path on the human holdout - if present it must be a measured run.
    live = REPO / "results" / "meta-llama-llama-3-3-70b-instruct-turbo.json"
    if live.exists():
        d = json.loads(live.read_text(encoding="utf-8"))
        assert d["status"] == "measured", d["status"]
        assert d["stimuli"].endswith("vignettes_test_human.jsonl"), d["stimuli"]


def test_default_stimuli_is_holdout():
    src = (REPO / "scoring" / "run_eval.py").read_text(encoding="utf-8")
    assert 'default="vignettes_test_human.jsonl"' in src


def test_no_structural_zero_in_public_results():
    for path in (REPO / "results").glob("*.json"):
        if path.name == "index.json":
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        st = d.get("subscore_status") or {}
        assert st.get("acoustic_risk_f1") != "structural_zero", path.name
        assert d.get("n_subscores_total") == 2, path.name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all passed")

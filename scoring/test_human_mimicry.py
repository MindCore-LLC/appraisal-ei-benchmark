"""human_mimicry checks. Run: python scoring/test_human_mimicry.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_mimicry
import score

REPO = Path(__file__).resolve().parents[1]


def _synthetic(model: str, n_items: int = 40, n_raters: int = 4, seed: int = 0):
    """Fake world: item effect + N(0,1) rater noise, on the real 17 dims.

    model='rater'     fresh draw from the same distribution as a rater
    model='smeared'   deviations shrunk 3x toward consensus
    model='echo'      outputs the consensus exactly (zero deviation)
    """
    rng = np.random.default_rng(seed)
    dims = score.load_dimensions()
    effect = rng.uniform(-2, 2, (n_items, len(dims)))
    by_item: dict[str, list[dict]] = {}
    for i in range(n_items):
        item_id = f"syn_{i:03d}"
        by_item[item_id] = [
            {
                "id": item_id,
                "rater_id": f"rater_{r}",
                "ratings": {d: float(np.clip(effect[i, k] + rng.normal(0, 1.0), -3, 3))
                            for k, d in enumerate(dims)},
            }
            for r in range(n_raters)
        ]
    model_items: dict[str, dict[str, float]] = {}
    for i in range(n_items):
        item_id = f"syn_{i:03d}"
        consensus = {d: float(np.mean([r["ratings"][d] for r in by_item[item_id]])) for d in dims}
        if model == "echo":
            model_items[item_id] = consensus
        elif model == "rater":
            # a genuinely exchangeable rater: draws around the item effect,
            # same generative process as the raters, not around the consensus
            model_items[item_id] = {
                d: float(np.clip(effect[i, k] + rng.normal(0, 1.0), -3, 3)) for k, d in enumerate(dims)
            }
        else:
            model_items[item_id] = {
                d: float(np.clip(consensus[d] + rng.normal(0, 1.0 / 3.0), -3, 3)) for d in dims
            }
    return by_item, model_items


def test_rank_auc_basics():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 1, 1])
    assert human_mimicry.rank_auc(x, y) == 1.0
    assert human_mimicry.rank_auc(-x, y) == 0.0
    assert human_mimicry.rank_auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert human_mimicry.rank_auc(np.array([1.0, 2.0]), np.array([1, 1])) is None


def test_rater_like_model_scores_high():
    by_item, model_items = _synthetic("rater")
    out = human_mimicry.mimicry(model_items, by_item)
    assert out is not None
    assert out["n_items"] == 40
    assert out["human_mimicry"] > 0.75, out


def test_smeared_model_is_distinguishable():
    by_item, model_items = _synthetic("smeared")
    out = human_mimicry.mimicry(model_items, by_item)
    assert out is not None
    assert out["human_mimicry"] < 0.5, out


def test_consensus_echo_is_caught():
    by_item, model_items = _synthetic("echo")
    out = human_mimicry.mimicry(model_items, by_item)
    assert out is not None
    assert out["human_mimicry"] < 0.2, out


def test_insufficient_data_is_none():
    by_item, model_items = _synthetic("rater", n_items=5)
    assert human_mimicry.mimicry(model_items, by_item) is None


def test_deterministic():
    by_item, model_items = _synthetic("rater", seed=7)
    assert human_mimicry.mimicry(model_items, by_item) == \
        human_mimicry.mimicry(model_items, by_item)


def test_real_results_measured():
    rater_by_item = human_mimicry.human_ceiling.load_rater_file()
    d = json.loads((REPO / "results" / "gpt-5-4.json").read_text(encoding="utf-8"))
    out = human_mimicry.mimicry(human_mimicry.load_model_items(REPO / "results" / "gpt-5-4.json"),
                                rater_by_item)
    assert out is not None
    assert out["n_items"] == d["n_items"] == 84
    assert 0.0 <= out["human_mimicry"] <= 1.0


def test_human_reference_measured():
    """Real raters are NOT at mimicry 1.0 vs the pool - each has a fingerprint
    the discriminator learns. The reference band is the interpretive context
    for model scores (see SPEC section 3), so it must be computable and
    deterministic, not necessarily high."""
    rater_by_item = human_mimicry.human_ceiling.load_rater_file()
    ref = human_mimicry.human_reference(rater_by_item)
    assert set(ref) == {"rater_1", "rater_2", "rater_3", "rater_4"}
    assert all(m is not None and 0.0 <= m <= 1.0 for m in ref.values()), ref
    assert ref == human_mimicry.human_reference(rater_by_item)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all passed")

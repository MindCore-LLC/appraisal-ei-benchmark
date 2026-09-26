"""Self-check for the 1.1.0 scoring changes. Run: python scoring/test_score.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score


def test_unmeasured_is_excluded_not_zero():
    one = {"appraisal_tracking": 0.42}
    assert score.aggregate_ei(one) == 0.42, "a lone public metric must not be diluted"
    assert score.aggregate_ei({}) is None, "nothing measured must be None, not 0.0"
    # v2.0.0: a structural zero is recorded in status, never averaged in
    both = score.aggregate_ei(one, ("acoustic_risk_f1",))
    assert abs(both - 0.42) < 1e-9, both
    st = score.measured_subscores(one, ("acoustic_risk_f1",))
    assert st["appraisal_tracking"] == "measured"
    assert st["acoustic_risk_f1"] == "structural_zero"
    assert st["human_mimicry"] == "not_measured"


def test_aggregate_is_public_metrics_only():
    parts = {"appraisal_tracking": 0.80, "discriminant_validity": 0.20, "value_action": 0.99,
             "appraisal_calibration": 0.99}
    got = score.aggregate_ei(parts)
    assert abs(got - 0.50) < 1e-9, got
    assert "appraisal_tracking" in score.PUBLIC_KEYS
    assert "appraisal_calibration" not in score.PUBLIC_KEYS
    assert "value_action" not in score.PUBLIC_KEYS


def test_tracking_scores_constant_as_zero():
    rnd = random.Random(1)
    dims = [f"d{i}" for i in range(5)]
    gold = {d: [rnd.uniform(-3, 3) for _ in range(20)] for d in dims}
    const = {d: [1.0] * 20 for d in dims}
    assert score.appraisal_tracking(const, gold) == 0.0
    good = {d: [v + rnd.gauss(0, 0.3) for v in gold[d]] for d in dims}
    assert score.appraisal_tracking(good, gold) > 0.9
    # gold with no variance is unmeasurable, not a zero
    flat_gold = {d: [0.0] * 20 for d in dims}
    assert score.appraisal_tracking(good, flat_gold) is None


def test_discriminant_validity_separates_specific_from_smeared():
    rnd = random.Random(0)
    dims = [f"d{i}" for i in range(6)]
    gold = {d: [rnd.uniform(-3, 3) for _ in range(40)] for d in dims}

    # a model that tracks each dimension specifically
    specific = {d: [v + rnd.gauss(0, 0.3) for v in gold[d]] for d in dims}
    # a model emitting one valence signal for every dimension
    smear = {d: [v + rnd.gauss(0, 0.3) for v in gold[dims[0]]] for d in dims}

    hi = score.discriminant_validity(specific, gold)
    lo = score.discriminant_validity(smear, gold)
    assert hi is not None and lo is not None
    assert hi > 0.5, f"specific model should score high, got {hi}"
    assert lo < 0.2, f"smeared model should score near zero, got {lo}"
    assert hi > lo


def test_discriminant_validity_needs_data():
    assert score.discriminant_validity({}, {}) is None
    tiny = {f"d{i}": [1.0, 2.0, 3.0] for i in range(6)}
    assert score.discriminant_validity(tiny, tiny) is None, "too few items must be None"
    flat = {f"d{i}": [1.0] * 40 for i in range(6)}
    assert score.discriminant_validity(flat, flat) is None, "no variance must be None"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all passed")

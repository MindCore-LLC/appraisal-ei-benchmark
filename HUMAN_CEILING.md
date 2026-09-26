# Human ceiling

Each rater scored exactly the way a model is: against the mean of the *other*
raters (leave-one-rater-out), on the frozen holdout
`stimuli/text/vignettes_test_human.jsonl`.

| | |
|---|---|
| **Tracking (headline, v3.0.0)** | **0.6096** (per rater: `{"rater_2": 0.5673, "rater_3": 0.597, "rater_4": 0.6646}`) |
| Discriminant validity | 0.239 |
| Calibration (diagnostic only) | 0.8269 |
| human_mimicry band | [0.0555, 0.2174] (per rater: `{"rater_1": 0.4297, "rater_2": 0.1159, "rater_3": 0.2174, "rater_4": 0.0555}`) |
| Items with ≥2 raters | 84 |
| Raters | 4 |
| Pairwise unweighted κ | 0.165 |

Per-rater mean LOO calibration r: `{"rater_1": 0.7227, "rater_2": 0.8118, "rater_3": 0.8387, "rater_4": 0.8485}`

The tracking ceiling and the mimicry band use raters who rated at least 80% of
the holdout. rater_1 rated 16 of 84 holdout items, so it is listed but does
not set either number.

## Why calibration is no longer the headline

Calibration is the per-item correlation across the 17 dims. Most of it is the
typical appraisal profile, which every situation shares. A constant average
profile scores 0.864 on it, above this ceiling. Tracking asks whether a
rating moves with the humans' when the situation changes, per dimension; the
same constant scores 0. See the baseline rows on the board.

## Reading human_mimicry

Publish the band next to every score. A model inside the band is as
distinguishable from the rater pool as a real rater is. It is not proof of
human-likeness: the average profile plus random noise at the raters' spread
lands inside the band too (`results/baseline-average-noise.json`).

## Why leave-one-out

A model is scored against the consensus of the raters. Scoring a rater against
a consensus they helped create inflates the ceiling. LOO is the comparable
number: each rater is treated the way a model is treated.

Do **not** claim a model "beats humans" if its calibration vs full consensus sits
next to this LOO number. Those are neighbouring measurements, not the same
test. The honest sentence is: frontier models are in the same band as a
held-out rater.

Unweighted pairwise κ on a 7-point scale is expected to look low; adjacent-point
disagreement is common. The scoring metric is Pearson r, so the ceiling is r.

## Recompute

```
python scoring/human_ceiling.py
python scripts/pack_human_gold.py
python scripts/backfill_results.py   # re-adds v3 fields to results/human.json
```

Source vectors: `annotation/rater_vectors.jsonl` (anonymized `rater_N` ids).
Round-3 emails never enter this repo.

## Unique scenarios

`stimuli/text/vignettes_unique_human.jsonl` collapses name-swapped duplicates
to **94** distinct texts (from 560 rated rows). The
84-item holdout covers **23** of those
unique scenarios. Public rank still uses the 84 rows so published model scores
stay comparable. Do not quote the 560-row file as 560 scenarios.

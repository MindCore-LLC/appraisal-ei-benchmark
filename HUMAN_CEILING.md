# Human ceiling

Leave-one-rater-out Pearson r against the mean of the other raters, on the
frozen holdout `stimuli/text/vignettes_test_human.jsonl`.

| | |
|---|---|
| **Calibration (headline)** | **0.821** |
| Discriminant validity | 0.239 |
| Items with ≥2 raters | 84 |
| Raters | 4 |
| Pairwise unweighted κ | 0.165 |

Per-rater mean LOO r: `{"rater_1": 0.7246, "rater_2": 0.8093, "rater_3": 0.8333, "rater_4": 0.8381}`

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
```

Source vectors: `annotation/rater_vectors.jsonl` (anonymized `rater_N` ids).
Round-3 emails never enter this repo.

## Unique scenarios

`stimuli/text/vignettes_unique_human.jsonl` collapses name-swapped duplicates
to **94** distinct texts (from 560 rated rows). The
84-item holdout covers **23** of those
unique scenarios. Public rank still uses the 84 rows so published model scores
stay comparable. Do not quote the 560-row file as 560 scenarios.

# Appraisal-EI Benchmark - Specification v3.0.0

Emotional-intelligence benchmark measuring **appraisal structure** - whether a
model represents *why* someone feels a way (17 appraisal dimensions), not just
*what* they feel (emotion labels). This is complementary to EQ-Bench
(emotional reasoning QA) and EmoBench (emotion recognition). The scored object
is a 17-dimensional appraisal vector compared to human gold ratings.

Taxonomy: Smith & Ellsworth; Scherer's Component Process Model; OCC.

Pin a **git tag**. Never evaluate against `main`.

## Design principles (steal these)

1. Score the structure that produces emotion, not the emotion label.
2. Human gold, or the run is `smoke` and does not enter the main board.
3. Never average a score you did not measure. A missing arm is `null`, not 0.
4. Publish per-rater vectors, not only consensus.
5. Publish a human ceiling computed the same way a model is scored
   (leave-one-rater-out vs the others).
6. Pin a tag. Stimuli are append-only within a major version.
7. Document your own QC failures in-repo. Do not silently retire broken sets.
8. Every headline ships with a no-model baseline row. If a constant or random
   answer scores well, the metric is wrong, not the baseline.

## 1. Rating schema (`schema/rating_schema.json`)

- 17 dimensions, each scored on a continuous **-3.0 to +3.0** scale
- Every scored item produces a JSON object keyed by the 17 dimension ids
- `anticipated_emotion` is the 17th dimension (prospective affect), not a
  categorical label - the benchmark has no emotion-classification task

## 2. Stimuli

### Frozen public holdout (headline)

- `stimuli/text/vignettes_test_human.jsonl` - 84 rows, human consensus,
  3–4 calibrated core raters per item. This is the **only** set the public
  rank is computed on. Name-swapped duplicates mean these 84 rows are
  **23 unique scenarios**; quote both numbers. Confidence intervals resample
  scenarios, not rows.
- **Contamination rule.** 9 of the 23 holdout scenarios have name-swapped
  copies outside the holdout (104 item ids, 84 of them in the train split).
  Exact-text dedup does not catch them. `stimuli/text/holdout_scenario_twins.json`
  lists every one; a model trained on any of them has seen the test and its
  run is not a holdout result.

### Unique-scenario gold (not the rank set)

- `stimuli/text/vignettes_unique_human.jsonl` - 94 distinct texts after
  masking first names, from 560 rated rows. Use this when you need one
  vector per scenario. Do not advertise 560 scenarios.

### crowd-enVENT (mapping not yet validated against this rubric)

- `envent_{train,val,test}.jsonl` - 1,200 event descriptions with
  reader-consensus ratings mapped onto our 17 dims. `fairness` is unmapped.
  Treat as adjacent human gold until the mapping study lands (see GitHub
  issues). Not the headline set.

### Deprecated

- `vignettes_{train,val,test}.jsonl` - generator priors. 560 rows, 94 texts,
  train/test leakage, intensity never reached the text. See
  `stimuli/text/QC_FINDINGS.md`. Smoke only. Not on the main board.
- `results/archive/` - historical smoke runs, retained so history is not
  rewritten.

### Audio

- `stimuli/audio/manifest_v1.jsonl` - RAVDESS references. Wavs are not
  redistributed. No public audio score in v2.0.0.

## 3. Scoring protocol (`scoring/score.py`)

**Headline = `appraisal_tracking`:** for each of the 17 dimensions, Pearson r
between the model's ratings and the human gold across the holdout items; the
headline is the mean over dimensions. It asks: when the situation changes,
does the model's rating move the way the humans' does? A dimension the gold
does not vary on (or with fewer than 8 pairs) is not measurable and is
skipped; a dimension the *model* does not vary on scores 0, because a constant
carries no information. Reported with a 95% CI from 2,000 bootstrap resamples
of the 23 holdout scenarios (`score.tracking_ci`).

**Second public metric = `discriminant_validity`:** mean diagonal minus mean
absolute off-diagonal of the multitrait matrix r(pred[i], gold[j]). High =
dimension-specific appraisal. Near zero = one valence signal smeared across
all 17 dimensions.

**Diagnostic, not ranked: `appraisal_calibration`** (the v2 headline): mean
per-item Pearson r between the model's 17-dim vector and the gold vector.
Kept so v2 numbers stay comparable. It is dominated by the typical appraisal
profile that every situation shares: the `baseline-average-profile` row (one
constant vector for every item) scores 0.864, above every model and the human
ceiling (0.821). Do not rank or quote it as a headline.

**Baseline rows** (`scripts/make_baselines.py`, train-split raters only,
seeded): `baseline-average-profile` (the mean train consensus for every item)
and `baseline-average-noise` (that profile plus Gaussian noise at the raters'
per-dimension spread around their LOO consensus, rounded to the scale). They
are on the board so every number can be read against "no model at all".

**Roadmap, measured: `human_mimicry`** (`scoring/human_mimicry.py`). Can a
discriminator tell the model's vectors from rater vectors? For every holdout
item with both rater rows and a model prediction, each human row is centered
by its leave-one-rater-out consensus and the model row by the full consensus;
a closed-form Gaussian discriminant (diagonal covariance, no hyperparameters)
is trained leave-one-item-out to tell the two apart. The pooled out-of-fold
Mann-Whitney AUC becomes `human_mimicry = 2 * (1 - max(AUC, 1 - AUC))`:
1.0 = indistinguishable from the rater pool, 0.0 = fully separable. Read the
score against the **human reference band** - the same protocol run with each
real rater as the "model" - because every rater has a fingerprint the
discriminator learns; a model inside the band is behaviorally exchangeable
with an individual rater. Reported per model; never averaged into a headline.
**Every published score carries the band** (`human_mimicry_band` in each
results file, from `human_mimicry.human_band`: raters who rated >= 80% of the
holdout). Inside the band is necessary, not sufficient: the
`baseline-average-noise` row lands inside it, so a mimicry score alone is not
evidence that a model rates like a person.

| Slot | Status in v3.0.0 |
|---|---|
| `appraisal_tracking` | **public, headline** |
| `discriminant_validity` | **public** |
| `appraisal_calibration` | diagnostic (not ranked) |
| `value_action` | roadmap |
| `persistence` | roadmap |
| `acoustic_risk_f1` | roadmap (audio track) |
| `steering_score` | roadmap |
| `human_mimicry` | roadmap, **measured** (reported with the human band, not averaged) |

Roadmap slots without a definition, a test, and gold that can support them
stay `null`. They are not averaged into anything. A text-only
run does **not** receive a structural zero on `acoustic_risk_f1`.

Per-dimension results: a dim the gold does not carry, or one with fewer than
8 paired observations, reports `r: null` and `measured: false`, never `r: 0.0`.

Paired comparisons use Wilcoxon signed-rank (`paired_test`), minimum 8 pairs.

## 4. Human ceiling

Computed by `scoring/human_ceiling.py` from `annotation/rater_vectors.jsonl`.

Each rater is scored exactly like a model, against the mean of the **other**
raters (leave-one-rater-out). The tracking ceiling is the mean over raters who
rated >= 80% of the holdout (0.610 in v3.0.0). See `HUMAN_CEILING.md`.

Models are still scored against the full consensus. Do not read a model
slightly above the LOO ceiling as "superhuman."

## 5. Annotation protocol

- Every holdout item rated by **>= 2** annotators
- Gold = per-dimension mean across annotators
- Agreement is reported as quadratic-weighted kappa **and** Krippendorff α
  (interval). Gate: weighted κ ≥ 0.6 **or** α ≥ 0.8. Unweighted kappa is
  reported for transparency and is not the gate.
- Per-rater vectors are retained. Consensus-only gold is not sufficient.

## 6. Versioning

- `VERSION` + git tag pin the immutable artifact
- Stimuli are append-only within a major version
- Scoring-rule changes that alter published numbers are a major (2.0)
- v3.0.0 changes vs 2.x: headline is `appraisal_tracking` with a
  scenario-resampled CI; `appraisal_calibration` demoted to diagnostic
  because a constant profile beat the human ceiling on it; no-model baseline
  rows; human band published with every `human_mimicry`; holdout twin list
  and contamination rule. **Gold correction:** the round-3 rating form
  anchored `attribution` reversed relative to this schema (+3 = other
  people), so human attribution gold is negated to the schema's scale
  (`scripts/fix_attribution_convention.py`; rows marked
  `attribution_convention: schema`). Residual: circumstance, which the form
  put at 0, stays at 0 here (the schema puts it at -3). Schema 1.2.0 also
  rewrites `anticipated_emotion` to what raters rated: the valence of the
  later feeling, not change relative to now. Evidence: r(responsibility,
  attribution) across rater vectors was -0.80 before the fix, +0.79 after;
  frontier models had negative attribution r against the old gold.
- v2.0.0 changes vs 1.x: headline is calibration; structural zeros no longer
  enter any mean; smoke is archived off the main board; human ceiling is a
  first-class row

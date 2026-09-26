# Appraisal-EI Benchmark

Does a model represent *why* someone feels a way - a 17-dimensional appraisal
vector - rather than just *what* they feel.

Companion to [EQ-Bench](https://eqbench.com) and EmoBench, not a replacement.

**Spec: [SPEC.md](SPEC.md). Version: 3.0.0. Pin a tag, never `main`.**

## Headline numbers

Ranked on **appraisal tracking**: for each of the 17 dimensions, does the
model's rating move with the humans' across situations (Pearson r across
items), averaged over dimensions, with a 95% CI that resamples the 23 unique
holdout scenarios. Second public metric: **discriminant validity** (did the
model recover 17 dimensions, or smear valence?).

A **Human ceiling** row sits on the board: leave-one-rater-out tracking =
**0.610**. See [HUMAN_CEILING.md](HUMAN_CEILING.md). Two **no-model baseline**
rows (a constant average profile, and that profile plus random noise) sit on
it too, so every score can be read against "no model at all".

Why v3 changed the headline: the v2 headline (per-item calibration) was
topped by the constant average profile (0.864, above the human ceiling).
It is kept as a diagnostic. `human_mimicry` is published with the human band,
and the noise baseline shows that landing inside the band is not enough on
its own.

**Gold correction in v3:** human `attribution` gold was rated on a form
that reversed the schema's direction; it is now on the schema's scale, and
`anticipated_emotion` is defined as what raters actually rated (valence of
the later feeling). See SPEC.md versioning.

**Training on this benchmark's data:** never train on the ids in
`stimuli/text/holdout_scenario_twins.json` (name-swapped copies of holdout
scenarios).

```
python scoring/run_eval.py --provider openai --model gpt-5.4
python scoring/test_score.py
python site/build_site.py
```

Default stimuli are the human-gold holdout. Generator-prior files exist for
history and score as `smoke`; they do not enter the main board.

## Layout

```
schema/rating_schema.json              # 17 dims, -3..+3 (normative)
stimuli/text/vignettes_test_human.jsonl  # frozen 84-item holdout
stimuli/text/vignettes_unique_human.jsonl # 94 distinct scenarios
stimuli/text/QC_FINDINGS.md            # why 560 ≠ 560
annotation/rater_vectors.jsonl         # per-rater gold, anonymized
scoring/score.py                       # reference scorer
scoring/human_ceiling.py               # recompute the human row
HUMAN_CEILING.md
CONTRIBUTING.md                        # eval contract
```

## Status

v3.0.0 - headline is tracking; calibration is diagnostic. Public metrics
are tracking + discriminant validity. Roadmap arms (audio, value–action, persistence, steering) are
named and null; `human_mimicry` is measured and reported per model, never
averaged into a headline. crowd-enVENT mapping onto this rubric is not yet
validated; do not lead with it.

## License

Code and original stimuli: MIT. RAVDESS audio is **not** redistributed.

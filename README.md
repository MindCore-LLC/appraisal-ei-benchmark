# Appraisal-EI Benchmark

Does a model represent *why* someone feels a way — a 17-dimensional appraisal
vector — rather than just *what* they feel.

Companion to [EQ-Bench](https://eqbench.com) and EmoBench, not a replacement.

**Spec: [SPEC.md](SPEC.md). Version: 2.0.0. Pin a tag, never `main`.**

## Headline numbers

Ranked on **appraisal calibration** (mean per-item Pearson r vs human gold)
on the frozen 84-item holdout (`vignettes_test_human.jsonl` — 23 unique
scenarios). Second public metric: **discriminant validity** (did the model
recover 17 dimensions, or smear valence?).

A **Human ceiling** row sits on the board: leave-one-rater-out r = **0.821**.
See [HUMAN_CEILING.md](HUMAN_CEILING.md).

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

v2.0.0 — protocol freeze. Public metrics are calibration + discriminant
validity. Roadmap arms (audio, value–action, persistence, steering) are
named and null; `human_mimicry` is measured and reported per model, never
averaged into a headline. crowd-enVENT mapping onto this rubric is not yet
validated; do not lead with it.

## License

Code and original stimuli: MIT. RAVDESS audio is **not** redistributed.

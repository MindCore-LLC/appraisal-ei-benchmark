# Stimulus QC: vignettes_*.jsonl (v1.2.0, 560 items)

Found while polishing the rating schema, 2026-09-19. Each item below was measured
against `stimuli/text/vignettes_{train,test,val}.jsonl`, not estimated.
Items 1-3 block the Prolific study; items 4-5 affect scoring.

## 1. The corpus is 94 scenarios, not 560

Masking first names, the 560 vignettes collapse to **94 distinct texts**. The
usual pattern is 6 vignettes per scenario: 2 names x 3 intensity levels. Largest
cluster is 17. Only 36 scenarios are singletons.

Consequence: five raters x 560 items buys ~94 scenarios of signal and spends the
rest re-rating sentences a rater has already seen. It also inflates apparent
inter-rater agreement, because repeat exposure to identical text is not an
independent observation.

## 2. `intensity` never reached the text

Each scenario is emitted at intensity 1.0, 0.7 and 0.4, and **the text is
byte-identical across all three** (verified on all 58 multi-level clusters). The
level survives only as metadata.

The generator priors nevertheless shift with it, on identical text:

| dimension | mean prior shift, 1.0 vs 0.4 |
|---|---|
| goal_relevance | +0.57 |
| effort | +0.50 |
| novelty | +0.43 |
| urgency | +0.36 |
| pleasantness | -0.26 |

So the labels encode a distinction that is absent from the stimulus. No rater and
no model can recover it. Any model scored on these items is penalised for failing
to read information that was never written down. This is the most serious defect
here: it makes ~58 of 94 scenarios unscoreable as designed.

Name-only pairs behave correctly by contrast: all 121 same-scenario,
same-intensity pairs carry identical priors.

## 3. Train/test leakage

15 scenarios appear in more than one split; **9 put a test item and a train or
val item on identical text**. 216 vignettes sit in a split-spanning cluster. A
model fitted on train has seen the test sentence verbatim.

## 4. `attribution` carries zero information

`attribution` is **0.0 on all 560 items** - no variance. Correlation-based
scoring against a constant is undefined. This is the same dimension whose prompt
asks a three-way question ("them, others, or circumstances?") on a two-way scale;
the schema now records that limitation, but the prior data suggests the generator
simply could not answer it either.

## 5. Priors are mostly zero, and half of several scales is unused

Over 80% zeros: `goal_congruence`, `fairness`, `expectation`, `coping_potential`,
`social_desirability`, `moral_worth`, `attribution`, `anticipated_emotion`.

`goal_congruence` at 488/560 zeros is hard to credit for a corpus of job losses,
betrayals and breakups - `vig_0000` is a layoff with `pleasantness -2.0` and
`goal_congruence 0.0`. 12 of 17 dimensions never use one side of their range.

These are generator priors, explicitly `smoke` status, so this does not
invalidate published numbers. It does mean the priors are a weak sanity baseline
and should not be used to pre-fill or anchor the human study.

## Decisions needed before recruiting raters

- **Dedup or differentiate.** Either cut to ~94 scenarios, or rewrite so the
  intensity levels actually differ in text. Cutting is cheaper; rewriting keeps
  the intensity axis, which appears to be a deliberate design variable.
- **Re-split after dedup**, grouping by scenario so no scenario spans splits.
- **`attribution`**: split into two dimensions, or narrow the prompt to the
  external/internal axis the scale can express. Changing `n_dimensions` breaks
  `scoring/score.py` and every stored result, so this is a versioned change.

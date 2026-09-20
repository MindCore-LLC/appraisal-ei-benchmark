# Human annotation: revised plan (v2, 2026-09-19)

Supersedes the original scope, which was "rate all 560 vignettes to turn them
into human gold." That scope is withdrawn. Reasons in
`stimuli/text/QC_FINDINGS.md`: the 560 rows carry 94 distinct texts, and the
intensity variable never reached the text, so most of the budget would buy
re-ratings of sentences a rater had already seen.

The repo already contains human gold: `envent_*.jsonl`, 1200 items, 1198
distinct texts, five readers each, from crowd-enVENT (Troiano, Oberlander &
Klinger 2023, CL 49(1)). As of `gold_status.json` 1.4.0 that is the primary
evaluation set. Annotation spend should therefore buy what envent cannot give
us, in this order.

## Priority 1: validate the envent to 17-dim mapping (highest value)

The single methodological soft spot in leading with envent is that the mapping
from its SEC variables onto our 17 dimensions is **ours, not theirs**, and it has
never been checked against a human using our wording.

- Sample ~120 envent items, stratified by `family` and by `emotion_author`.
- Rate them with our own rubric and anchors (`annotation/label_config.xml`), by
  >= 3 raters per item.
- Report the correlation between our raters' 17-dim vectors and the mapped
  `ratings` already in the file, per dimension.
- Any dimension where our raters and the mapping disagree is a dimension we
  cannot report against envent gold. Expect trouble on `fairness`, which has no
  crowd-enVENT analogue and is currently unmapped.

This converts "we reinterpreted someone else's corpus" into a measured claim,
for roughly a fifth of the original budget.

## Priority 2: retain per-annotator vectors, always

`human_mimicry` is blocked solely because crowd-enVENT ships consensus rather
than individual raters (`annotators` is a count). The same is true of
`pairwise_kappa`, so the `kappa > 0.6` bar in SPEC section 4 is currently
unverifiable.

Every round from here stores **each rater's full 17-dim vector**, not just the
mean. This is a storage decision, not a budget decision, and it unblocks a
second sub-score for free.

## Priority 3: new scenarios, only if family-level claims are needed

The deduped diagnostic set averages 4.7 scenarios per family, and 13 of 20
families have exactly 3. No per-family claim survives that. If family-level
breakdowns are part of the story, commission **newly written scenarios**,
15-20 per family.

Name swaps add no information. Do not pad with them. If an intensity axis is
wanted, the intensity must be visible in the text, and that should be verified
by a manipulation check (raters rank the three levels correctly) before the
main round.

## Not planned

Re-rating the 94 diagnostic vignettes. They stay `generator_priors` and
`status=smoke`. They are a controlled probe for dimension-specific behaviour,
not a gold set, and human-rating them would not change what they are for.

## Instrument

`annotation/label_config.xml`, generated from `schema/rating_schema.json` by
`scripts/make_label_config.py`. Every dimension shows -3 / 0 / +3 anchor text.
Before any paid round:

- [ ] Disable or reassign Label Studio's auto-assigned hotkeys. They currently
      run `[1]`-`[7]`, then `[8][9][0][q][w][e][t]`, then `[a][s][d][f][g][z][x]`,
      and run out entirely after the fifth question.
- [ ] Decide `attribution`: split into two dimensions or narrow the prompt. It
      asks a three-way question on a two-way scale and is constant 0.0 across
      all 560 generator priors, which suggests the generator could not answer it
      either. Changing `n_dimensions` is a versioned break.
- [ ] Run 5-10 raters through a pilot and check per-dimension variance before
      committing the full sample.

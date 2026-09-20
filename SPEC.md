# Appraisal-EI Benchmark - Specification v1.0.0

Emotional-intelligence benchmark measuring **appraisal structure** - whether a
model represents *why* someone feels a way (17 appraisal dimensions), not just
*what* they feel (emotion labels). This distinguishes it from EQ-Bench
(emotional reasoning QA) and EmoBench (emotion recognition) - here the scored
object is a 17-dimensional appraisal vector compared to human gold ratings.

Taxonomy: Smith & Ellsworth; Scherer's Component Process Model; OCC.

## 1. Rating schema (`schema/rating_schema.json`)

- 17 dimensions, each scored on a continuous **-3.0 to +3.0** scale
- Every scored item produces a JSON object keyed by the 17 dimension ids,
  values numeric
- `anticipated_emotion` is the 17th dimension (prospective affect), not a
  categorical label - the benchmark deliberately has no emotion-classification
  task

## 2. Stimuli

### Text (`stimuli/text/`)

- `vignettes_train.jsonl` (392), `vignettes_val.jsonl` (84),
  `vignettes_test.jsonl` (84) - 560 scenario vignettes across 20 situation
  families, generated to be **keyword-free**: no emotion words in the text, so
  ratings measure appraisal inference rather than sentiment lexicon matching.
  v1.2.0 appended 210 items so the FULL corpus can be human-annotated
  (prereg amended to annotate all items, not a subset); all pre-1.2.0 ids
  keep their original text AND split, so prior results remain comparable
- Each row: `{id, text, ratings, source, split}` - `ratings` in the shipped
  corpus are generator priors; human gold ratings replace them downstream
  (see §4). The `synthetic_sketch` source marks prior labels - they are NOT
  evaluation ground truth
- `envent_train.jsonl` (973), `envent_val.jsonl` (113),
  `envent_test.jsonl` (114) - 1,200 event descriptions from the crowd-enVENT
  corpus (Troiano, Oberlander & Klinger 2023), each with reader-consensus
  human appraisal ratings (5 annotators/item) mapped onto our 17-dim schema.
  Rows carry `source: "human"` and score as `measured`. Mapping and
  provenance: DATASETS.md. `fairness` is unmapped (no SEC analog);
  `anticipated_emotion` is author-grounded

### Audio (`stimuli/audio/`)

- `manifest_v1.jsonl` - 1,440 RAVDESS clips: `{id, stimulus, corpus, wav_ref,
  emotion, intensity, speaker_id, risk_level, risk_positive, source}`
- `wav_ref` is the canonical `Actor_XX/<clip>.wav` path in a standard RAVDESS
  download. **The wavs are not redistributed** (RAVDESS is research-license);
  consumers run the download script or supply their own copy
- `emotion`/`intensity`/`risk_*` are corpus metadata for stratification and QC,
  not the rating target - the rating target is the same 17-dim appraisal
  vector scored from the *vocal delivery*

## 3. Scoring protocol (`scoring/score.py`)

Per item: model emits a 17-dim vector; score = **Pearson correlation against
the human gold vector** for that item (`ratings_correlation`).

Aggregate EI = unweighted mean of the sub-scores **that were measured on that
run**. A sub-score that is not implemented, or that the stimuli cannot support,
is reported as `null` and excluded from the mean. It is never averaged in as 0:
a zero is a claim about the model, and we do not make claims we did not measure.

Every result therefore carries `n_subscores_measured` / `n_subscores_total`, and
`aggregate_ei` must always be quoted with that ratio. A mean over two sub-scores
is not comparable to a mean over seven.

| Sub-score | Status | Source |
|---|---|---|
| `appraisal_calibration` | **implemented** | mean per-item Pearson r vs gold |
| `discriminant_validity` | **implemented** | mean diagonal minus mean absolute off-diagonal of the multitrait matrix r(pred[i], gold[j]) across items. High = dimension-specific appraisal; near zero or negative = one valence signal smeared across all 17 dimensions |
| `acoustic_risk_f1` | structural zero (text runs) | vocal-risk detection (audio arm) |
| `value_action` | not implemented | stated appraisal vs chosen action consistency |
| `persistence` | not implemented | appraisal-state stability across turns |
| `steering_score` | not implemented | predicted direction of appraisal steering |
| `human_mimicry` | **blocked** | human-vs-model output discrimination. Requires per-annotator rating vectors. The crowd-enVENT release supplies reader *consensus* only (`annotators` is a count), so this cannot be computed from current stimuli and needs an annotation round that retains individual raters |

Text-only baselines score 0 on `acoustic_risk_f1` by construction; this is a
documented structural zero, not a missing measurement. The distinction matters:
structural zeros are averaged in, unmeasured sub-scores are not.

Per-dimension results follow the same rule. A dimension the gold does not carry
(`fairness` has no crowd-enVENT analogue) or one with fewer than 8 paired
observations reports `r: null` and `measured: false`, never `r: 0.0`.

Paired comparisons use **Wilcoxon signed-rank** (`paired_test`) with a
minimum of 8 paired observations.

## 4. Annotation protocol

- Every item rated by **>= 2 annotators**; inter-annotator agreement =
  mean pairwise Cohen's kappa across dimensions, target **kappa > 0.6**
- Export formats are in `annotation/`: `prolific_export_text.csv` (text
  ratings task) and `prolific_export_audio.csv` (audio ratings task, 296-clip
  stratified subset of the manifest). `*.prompts.json` = the per-dimension
  rating prompts and scale shown to annotators
- Gold ratings = per-dimension mean across annotators; only items meeting
  the agreement bar count as evaluation ground truth

## 5. Versioning

- `VERSION` + git tag pin the immutable artifact. Consumers reference a
  **tag**, never a branch
- Stimuli are append-only within a major version: corrections land in a new
  minor (1.0 -> 1.1); schema changes are a major (2.0)
- Generator priors and annotator exports are reproducible artifacts, not
  ground truth - the truth set is versioned separately once human ratings
  land (planned `gold/` directory, v1.1)

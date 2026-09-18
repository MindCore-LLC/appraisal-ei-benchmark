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

- `vignettes_train.jsonl` (244), `vignettes_val.jsonl` (52),
  `vignettes_test.jsonl` (54) - 350 scenario vignettes across 20 situation
  families, generated to be **keyword-free**: no emotion words in the text, so
  ratings measure appraisal inference rather than sentiment lexicon matching
- Each row: `{id, text, ratings, source, split}` - `ratings` in the shipped
  corpus are generator priors; human gold ratings replace them downstream
  (see §4). The `synthetic_sketch` source marks prior labels - they are NOT
  evaluation ground truth

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

Aggregate EI = unweighted mean of seven sub-scores:

| Sub-score | Source |
|---|---|
| `appraisal_calibration` | mean per-item Pearson r vs gold |
| `value_action` | stated appraisal vs chosen action consistency |
| `persistence` | appraisal-state stability across turns |
| `acoustic_risk_f1` | vocal-risk detection (audio arm) |
| `steering_score` | predicted direction of appraisal steering |
| `discriminant_validity` | appraisal correlation vs unrelated constructs |
| `human_mimicry` | human-vs-model output discrimination (lower = better mimicry gap closed) |

Text-only baselines score 0 on `acoustic_risk_f1` by construction; this is a
documented structural zero, not a missing measurement.

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

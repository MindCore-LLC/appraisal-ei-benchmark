# Appraisal-EI Benchmark

A benchmark for **appraisal structure** in language models: does a model
represent *why* someone feels a way - a 17-dimensional appraisal vector -
rather than just *what* they feel.

Companion to [EQ-Bench](https://eqbench.com) and EmoBench, not a replacement:
those test emotional reasoning and recognition; this tests the appraisal
dimensions that appraisal theory says *produce* the emotion.

**Spec: [SPEC.md](SPEC.md) - the normative document. Version: 1.5.0.**

## Layout

```
schema/rating_schema.json     # 17 dims, prompts, -3..+3 scale (normative)
stimuli/text/                 # 560 template vignettes + 2,038 round-4 AI vignettes
                              # + 84 human-gold test items (vignettes_test_human.jsonl)
                              # + 1,200 crowd-enVENT human-rated events
stimuli/audio/manifest_v1     # 1,440 RAVDESS refs (wavs not redistributed)
annotation/                   # Prolific/MTurk task exports (text + audio)
scoring/score.py              # reference scorer (numpy+scipy only)
scripts/                      # RAVDESS download helper
DATASETS.md                   # external corpus provenance, licenses, dim mapping
```

## Consuming

Pin a tag. Never float on a branch.

- Text stimuli: `stimuli/text/vignettes_{train,val,test}.jsonl` (generator
  priors - `smoke`) and `stimuli/text/envent_{train,val,test}.jsonl`
  (crowd-enVENT reader-consensus human gold - `measured`)
- Audio stimuli: `stimuli/audio/manifest_v1.jsonl` + your own RAVDESS
  download (`scripts/fetch_ravdess.py` maps `wav_ref`)
- Rating schema: `schema/rating_schema.json`
- Scoring: `scoring/score.py` - `aggregate_ei`, `ratings_correlation`,
  `pairwise_kappa`, `paired_test` (Wilcoxon). Stdlib + numpy + scipy only
- Runner: `scoring/run_eval.py --stimuli <file>` selects the stimulus file;
  `gold_status.json` declares provenance per file

## Status

v1.0.0 - stimuli + schema + scorer frozen. The envent_* files carry human
gold (crowd-enVENT, see DATASETS.md) and score as `measured`. MindCore
vignettes remain `generator_priors`/`smoke` until our own human annotation
study lands (see SPEC §4).

## License

Code and original stimuli: MIT (see LICENSE). RAVDESS audio is **not**
redistributed - `wav_ref` paths map to the official corpus, which is
research-licensed; get it from the source.

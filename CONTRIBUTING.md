# Contributing a model result

This is the eval contract. A result that breaks it is not admitted to the
main board.

1. Pin the git **tag** that matches `VERSION`. Do not score `main`.
2. Use `scoring/run_eval.py`. The rubric is generated from
   `schema/rating_schema.json`. Do not tune the prompt per model.
3. Default stimuli: `stimuli/text/vignettes_test_human.jsonl`.
   Generator-prior files (`vignettes_train/val/test.jsonl`) produce `smoke`
   and stay off the main board.
4. The reference scorer is `scoring/score.py`. Headline =
   `appraisal_calibration`. Discriminant validity is the second public
   metric. Unimplemented slots are `null`, never 0.
5. Result JSON lands in `results/<slug>.json` with `per_item` pred+gold
   vectors so anyone can recompute.
6. Add a `models.json` registry row (name, org, origin, open/closed).
7. Rebuild the site: `python site/build_site.py`.
8. Smoke, private gold, or a custom prompt will be rejected.

Human ceiling is recomputed from `annotation/rater_vectors.jsonl`, not
hand-edited. See `HUMAN_CEILING.md`.

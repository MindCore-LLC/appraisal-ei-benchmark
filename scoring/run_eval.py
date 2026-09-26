"""Run a model over Appraisal-EI stimuli and write results/<slug>.json.

Canonical protocol (SPEC.md section 3): zero-shot rubric prompt per item,
model returns a 17-dim JSON vector, per-item score = Pearson r vs gold.

    python scoring/run_eval.py --provider together --model deepseek-ai/DeepSeek-V4-Pro-0813
    python scoring/run_eval.py --provider openai --model gpt-5.4 --limit 20
    python scoring/run_eval.py --provider openai --model gpt-5.4 --stimuli envent_test.jsonl

The result records gold_source. Stimuli whose gold_status.json entry (or row
source fields) resolves to human* produce status "measured"; generator_priors
and anything else produce "smoke". Default stimuli are the v2 frozen holdout
vignettes_test_human.jsonl. Smoke results are not admitted to the main board.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import score  # scoring/score.py, same dir
import score_items  # scoring/score_items.py, same dir

REPO = Path(__file__).resolve().parents[1]
STIMULI_DIR = REPO / "stimuli" / "text"
SCHEMA = REPO / "schema" / "rating_schema.json"

BASE_URLS = {
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "anthropic": "https://api.anthropic.com/v1/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
}
PROVIDERS = sorted(KEY_ENV) + ["mock"]


def load_dimension_ids() -> list[str]:
    return [d["id"] for d in json.loads(SCHEMA.read_text(encoding="utf-8"))["dimensions"]]


def load_env(path: str | None) -> None:
    """Load KEY=VALUE lines from an env file without printing anything."""
    if not path:
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def rubric_prompt(vignette: str, dimensions: list[dict]) -> str:
    lines = [
        "Score the scenario on 17 appraisal dimensions from -3 to +3.",
        *[f"- {d['id']}: {d['prompt']}" for d in dimensions],
        "Return JSON only, keys exactly those 17 names, values numbers.",
        f"Scenario:\n{vignette}\nJSON:",
    ]
    return "\n".join(lines)


def generate(provider: str, model: str, prompt: str, max_tokens: int) -> tuple[str, dict | None]:
    """Run one prompt. Returns (text, usage) - usage is None when the provider
    does not report token counts (e.g. mock). Callers time the call for the
    runtime block; this function only extracts what the API already reports."""
    # `mock` exercises the whole pipeline - stimulus load, gold/status
    # resolution, parsing, scoring, output shape - without an API call, so a
    # paid run can be rehearsed first. Emits a deliberately poor but parseable
    # vector; the numbers are meaningless by design.
    if provider == "mock":
        import hashlib

        h = hashlib.sha256(prompt.encode()).digest()
        dims = load_dimension_ids()
        return json.dumps({d: (h[i] % 7) - 3 for i, d in enumerate(dims)}), None

    from openai import OpenAI

    kwargs: dict = {"api_key": os.environ.get(KEY_ENV[provider])}
    if provider in BASE_URLS:
        kwargs["base_url"] = BASE_URLS[provider]
    if not kwargs["api_key"]:
        raise SystemExit(f"{KEY_ENV[provider]} not set")
    params: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    # GPT-5/o-series want max_completion_tokens and reject temperature.
    if provider == "openai" and model.startswith(("gpt-5", "o1", "o3", "o4")):
        params.pop("temperature")
        params["max_completion_tokens"] = params.pop("max_tokens")
    # Claude 5 generation: adaptive thinking is always on, sampling params are rejected.
    if provider == "anthropic":
        params.pop("temperature", None)
    resp = OpenAI(**kwargs).chat.completions.create(**params)
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens} if u else None
    return resp.choices[0].message.content or "", usage


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=PROVIDERS,
                   help="'mock' rehearses the run with no API call")
    p.add_argument("--model", required=True)
    p.add_argument("--name", default=None, help="display name for the leaderboard")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="completion cap; reasoning models need headroom for thinking + JSON")
    p.add_argument("--env-file", default=None, help="optional KEY=VALUE file for API keys")
    p.add_argument("--stimuli", default="vignettes_test_human.jsonl",
                   help="file under stimuli/text/; default is the frozen human-gold holdout")
    args = p.parse_args()
    load_env(args.env_file)
    stimuli_path = STIMULI_DIR / args.stimuli

    dims = json.loads(SCHEMA.read_text(encoding="utf-8"))["dimensions"]
    dim_ids = [d["id"] for d in dims]
    rows = [json.loads(l) for l in stimuli_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    # Gold provenance: per-file declarations in gold_status.json (glob-matched),
    # then a top-level default, then per-row source fields as a fallback.
    gs_path = STIMULI_DIR / "gold_status.json"
    declared = None
    if gs_path.exists():
        gs = json.loads(gs_path.read_text(encoding="utf-8"))
        for pat, meta in (gs.get("files") or {}).items():
            if fnmatch.fnmatch(stimuli_path.name, pat):
                declared = meta.get("ratings_provenance")
                break
        declared = declared or gs.get("ratings_provenance")
    gold_sources = {r.get("source") for r in rows} - {None}
    gold = declared or ("human" if "human" in gold_sources else "generator_priors" if gold_sources == {"synthetic_sketch"} else "mixed")
    # A mock run emits hash noise - never let it present as measured, whatever
    # gold it was pointed at.
    status = "smoke" if args.provider == "mock" else ("measured" if gold.startswith("human") else "smoke")

    items, corrs, preds = [], [], []
    latencies_ms: list[float] = []
    usage_list: list[dict] = []
    for r in rows:
        t0 = time.perf_counter()
        text, usage = generate(args.provider, args.model, rubric_prompt(r["text"], dims), args.max_tokens)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if usage:
            usage_list.append(usage)
        pred = score.parse_ratings(text, dim_ids)
        c = score.ratings_correlation(pred, r.get("ratings") or {}, dim_ids) if len(pred) >= 8 else None
        items.append({"id": r["id"], "r": c, "pred": pred if len(pred) >= 8 else None, "gold": r.get("ratings")})
        if c is not None:
            corrs.append(c)

    # Runtime block: per-call wall time + token usage, feeding the speed axis
    # of the quality-vs-speed visuals. Median/p95 over per-item calls; tok/s
    # uses completion tokens over summed call time (no concurrency, so the
    # sum is the honest denominator). Mock runs time the local stub - the
    # numbers are pipeline-rehearsal artifacts, not model speed.
    runtime = None
    if latencies_ms:
        ls = sorted(latencies_ms)
        n = len(ls)
        median = ls[n // 2] if n % 2 else (ls[n // 2 - 1] + ls[n // 2]) / 2
        p95 = ls[min(n - 1, math.ceil(0.95 * n) - 1)]
        prompt_tok = sum(u["prompt_tokens"] for u in usage_list)
        completion_tok = sum(u["completion_tokens"] for u in usage_list)
        total_s = sum(ls) / 1000
        runtime = {
            "n_timed": n,
            "total_ms": round(sum(ls)),
            "median_ms": round(median, 1),
            "p95_ms": round(p95, 1),
            "mean_ms": round(sum(ls) / n, 1),
            "prompt_tokens": prompt_tok or None,
            "completion_tokens": completion_tok or None,
            "tokens_per_sec": round(completion_tok / total_s, 1) if completion_tok and total_s > 0 else None,
        }

    # v3.0.0: every score (tracking + scenario CI, discriminant validity,
    # diagnostic calibration, human_mimicry + human band) comes from
    # score_items, the same code the backfill uses. A dim the gold does not
    # carry emits r null, never 0.0.
    scored = score_items.score_items(items, dim_ids)

    slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
    result = {
        "model": args.name or args.model,
        "model_id": args.model,
        "provider": args.provider,
        "benchmark_version": (REPO / "VERSION").read_text().strip(),
        "stimuli": f"stimuli/text/{stimuli_path.name}",
        "gold_source": gold,
        "status": status,
        "run_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_items": len(rows),
        "n_parsed": len(corrs),
        "runtime": runtime,
        **scored,
        "per_item": items,
    }
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{slug}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # rebuild the manifest the leaderboard page reads
    files = sorted(f.name for f in out_dir.glob("*.json") if f.name != "index.json")
    (out_dir / "index.json").write_text(json.dumps({"results": files}, indent=2), encoding="utf-8")

    print(json.dumps({k: result[k] for k in (
        "model", "status", "n_parsed", "n_items", "appraisal_tracking", "appraisal_tracking_ci95",
        "n_subscores_measured", "n_subscores_total", "runtime")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

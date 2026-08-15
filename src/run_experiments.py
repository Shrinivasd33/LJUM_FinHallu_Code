"""
Phase 2 + Phase 3 orchestration: runs every (task instance x model x
condition) combination and saves raw outputs, per Chapter 3, Section 3.7.

  Phase 2: zero-shot and few-shot, across all 3 models, 3 repeated runs each
           (temperature=0, so repeats are a reproducibility check, not meant
           to introduce variance).
  Phase 3: RAG - only run for (model, task, condition) combinations that
           had at least one wrong/null answer in Phase 2 zero-shot, per the
           "for task-model combinations that hallucinated in Phase 2" rule.

THIS SCRIPT MAKES REAL, BILLED API CALLS (GPT-4 is pay-per-use; Gemini/Groq
free tiers have rate limits but are otherwise free). It will not run at all
until config/.env has real API keys - see config/.env.example.

Usage:
    python src/run_experiments.py --dimension NPI --models gpt-4 --condition zero_shot --limit 10
    python src/run_experiments.py --dimension NPI --models gpt-4,llama-3-70b,gemini-3.5-flash --condition all

Start with --limit 5-10 and one model to sanity-check cost/behaviour before
running the full 100-200 instances x 3 models x 3 conditions.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import get_model
from src.rag import build_retriever_for_task, augment_prompt
from src.fin_hallu_score import score_instance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 3  # Chapter 3.7: "three repeated runs at temperature=0"


def load_prompts(dimension: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{dimension}_prompts.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/preprocessing.py (and src/data_loading.py first) before this script."
        )
    return pd.read_parquet(path)


def result_path(dimension: str, model_name: str, condition: str) -> Path:
    return RESULTS_DIR / f"{dimension}__{model_name}__{condition}.jsonl"


def already_done_ids(dimension: str, model_name: str, condition: str) -> set[str]:
    """Resumability: if this script is interrupted (rate limit, crash,
    laptop sleep), re-running skips instances already recorded."""
    path = result_path(dimension, model_name, condition)
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["id"], rec["run_idx"]))
            except Exception:
                continue
    return done


def run_condition(
    df: pd.DataFrame, dimension: str, model_name: str, condition: str, limit: int | None = None
):
    """condition in {'zero_shot', 'few_shot', 'rag'}"""
    model = get_model(model_name)
    out_path = result_path(dimension, model_name, condition)
    done = already_done_ids(dimension, model_name, condition)

    rows = df if limit is None else df.head(limit)
    retrievers = {}  # cache per task to avoid rebuilding TF-IDF repeatedly

    with open(out_path, "a", encoding="utf-8") as f:
        for _, row in rows.iterrows():
            for run_idx in range(N_REPEATS):
                if (row["id"], run_idx) in done:
                    continue

                if condition == "zero_shot":
                    prompt = row["prompt_zero_shot"]
                elif condition == "few_shot":
                    prompt = row["prompt_few_shot"]
                elif condition == "rag":
                    if row["task"] not in retrievers:
                        retrievers[row["task"]] = build_retriever_for_task(df, row["task"], exclude_id=row["id"])
                    retrieved = retrievers[row["task"]].retrieve(row["query"], k=2)
                    prompt = augment_prompt(row["prompt_zero_shot"], retrieved)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                output = model.generate(prompt)
                record = {
                    "dimension": dimension,
                    "task": row["task"],
                    "id": row["id"],
                    "model": model_name,
                    # Which backend actually served this call (e.g. "deepinfra"
                    # vs "groq" for llama-3-70b) - added 2026-08-15 after
                    # discovering Groq and DeepInfra serve this model through
                    # two different proprietary quantization schemes, not a
                    # byte-identical checkpoint (Chapter 3, Section 3.6;
                    # Chapter 5, Section 5.8). Records written before this fix
                    # don't carry this field, which is itself disclosed as a
                    # provenance-tracking limitation.
                    "provider": getattr(model, "provider", None),
                    "model_id": getattr(model, "model_id", None),
                    "condition": condition,
                    "run_idx": run_idx,
                    "ground_truth": row["ground_truth"],
                    "model_output": output,
                    "answer_format": row["answer_format"],
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                logger.info(
                    "%s | %s | %s | %s (run %d): %s -> %s",
                    dimension, model_name, condition, row["id"], run_idx,
                    str(row["ground_truth"])[:40], str(output)[:60] if output else "NULL",
                )


def hallucinating_task_model_pairs(dimension: str, model_name: str) -> set[str]:
    """Which tasks had >=1 imperfect zero-shot answer for this model -
    Phase 3 RAG rule: only test RAG where it's actually needed. Uses the same
    format-aware scorer as Phase 4 (not a naive exact-string match, which
    would misfire on bio_sequence/entity_list/triplet_list answers where a
    partially-correct output is still meaningfully wrong)."""
    path = result_path(dimension, model_name, "zero_shot")
    if not path.exists():
        return set()
    df = pd.read_json(path, lines=True)
    df["f1"] = df.apply(
        lambda r: score_instance(r["answer_format"], r["model_output"], r["ground_truth"]).f1,
        axis=1,
    )
    wrong = df[df["f1"] < 1.0]
    return set(wrong["task"].unique().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", required=True, choices=["NPI", "TAI", "EFS", "CRI"])
    parser.add_argument("--models", required=True, help="comma-separated: gpt-4,llama-3-70b,gemini-3.5-flash")
    parser.add_argument("--condition", required=True, choices=["zero_shot", "few_shot", "rag", "all"])
    parser.add_argument("--limit", type=int, default=None, help="cap instances (use for cheap sanity checks)")
    args = parser.parse_args()

    df = load_prompts(args.dimension)
    if df.empty:
        logger.error("No preprocessed data for dimension %s - nothing to run.", args.dimension)
        return

    model_names = [m.strip() for m in args.models.split(",")]
    conditions = ["zero_shot", "few_shot", "rag"] if args.condition == "all" else [args.condition]

    for model_name in model_names:
        for condition in conditions:
            if condition == "rag":
                needed_tasks = hallucinating_task_model_pairs(args.dimension, model_name)
                if not needed_tasks:
                    logger.info(
                        "%s/%s: no zero-shot results found yet, or no hallucinations detected - "
                        "skipping RAG (run zero_shot first).", args.dimension, model_name
                    )
                    continue
                subset = df[df["task"].isin(needed_tasks)]
                run_condition(subset, args.dimension, model_name, condition, args.limit)
            else:
                run_condition(df, args.dimension, model_name, condition, args.limit)


if __name__ == "__main__":
    main()

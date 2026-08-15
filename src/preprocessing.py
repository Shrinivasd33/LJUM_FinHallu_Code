"""
Phase 1 (part 2): Data pre-processing pipeline.

Converts raw task DataFrames (from data_loading.py) into standardised prompt
records ready for model querying, per Chapter 3, Section 3.4 of the thesis:
  - task input formatting (per task type)
  - ground truth alignment (already done in data_loading via `ground_truth`)
  - deterministic sampling to the 100-200 instances/task-category target
    (Chapter 3, Section 3.7), using RANDOM_SEED for reproducibility

Output: one row per (task, instance) with columns:
  dimension, task, id, prompt_zero_shot, prompt_few_shot, ground_truth, context
"""
import logging
import random
from pathlib import Path

import pandas as pd


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.task_config import INSTANCES_PER_TASK, RANDOM_SEED, PHRASEBANK_INSTRUCTION

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Design note (2026-07-19): FinBen's native `query` field for every task
# already contains the complete, self-sufficient instruction (exact tag
# vocabulary, format spec, and a ready-to-complete "Text/Context: ...\n
# Answer:" structure) - it is more precise and authoritative than any custom
# instruction layered on top would be. So prompts are built directly from
# `query`, plus one generic persona preamble. PhraseBank is the one
# exception: it's raw "sentence@label" data with no native instruction, so
# it gets a small task-specific instruction of its own.
GENERIC_PREAMBLE = (
    "You are a careful, precise financial analyst. Complete the task below "
    "exactly as instructed, following the required answer format precisely."
)


def _full_query(task: str, query: str) -> str:
    if task == "PhraseBank":
        return f"{PHRASEBANK_INSTRUCTION}\nText: {query}\nAnswer:"
    return query


def _build_zero_shot_prompt(task: str, query: str) -> str:
    return f"{GENERIC_PREAMBLE}\n\n{_full_query(task, query)}"


def _build_few_shot_prompt(task: str, query: str, exemplars: list[dict]) -> str:
    parts = [GENERIC_PREAMBLE, "\nHere are some worked examples:"]
    for ex in exemplars:
        parts.append(f"\n{_full_query(task, ex['query'])} {ex['ground_truth']}")
    parts.append("\nNow complete this one:")
    parts.append(_full_query(task, query))
    return "\n".join(parts)


def sample_task_instances(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministically sample up to `n` rows per task, keeping the full set
    if fewer than `n` are available (Chapter 3, Section 3.7 rule)."""
    rng = random.Random(seed)
    sampled = []
    for task_name, group in df.groupby("task"):
        if len(group) <= n:
            sampled.append(group)
        else:
            idx = rng.sample(list(group.index), n)
            sampled.append(group.loc[idx])
    return pd.concat(sampled, ignore_index=True) if sampled else df


def preprocess_dimension(
    df: pd.DataFrame, dimension: str, n_per_task: int = INSTANCES_PER_TASK, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    if df.empty:
        logger.warning("preprocess_dimension: empty input for %s, nothing to do", dimension)
        return df

    df = sample_task_instances(df, n_per_task, seed)

    # Build few-shot exemplar pool per task (drawn from the same task, excluding
    # the instance itself at prompt-build time to avoid leakage).
    rng = random.Random(seed)
    exemplar_pool = {}
    for task_name, group in df.groupby("task"):
        pool_size = min(5, len(group))
        exemplar_pool[task_name] = group.sample(n=pool_size, random_state=seed).to_dict("records")

    zero_shot_prompts, few_shot_prompts = [], []
    for _, row in df.iterrows():
        zero_shot_prompts.append(_build_zero_shot_prompt(row["task"], row["query"]))

        candidates = [
            e for e in exemplar_pool[row["task"]] if e["id"] != row["id"]
        ]
        k = min(4, len(candidates))  # FEW_SHOT_EXAMPLES target (3-5)
        exemplars = rng.sample(candidates, k) if k > 0 else []
        few_shot_prompts.append(
            _build_few_shot_prompt(row["task"], row["query"], exemplars)
        )

    df = df.copy()
    df["prompt_zero_shot"] = zero_shot_prompts
    df["prompt_few_shot"] = few_shot_prompts
    return df


def preprocess_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    processed = {}
    for dimension, df in raw.items():
        processed[dimension] = preprocess_dimension(df, dimension)
        out_path = PROCESSED_DIR / f"{dimension}_prompts.parquet"
        processed[dimension].to_parquet(out_path, index=False)
        logger.info("%s: %d instances preprocessed -> %s", dimension, len(processed[dimension]), out_path)
    return processed


if __name__ == "__main__":
    raw = {}
    for dim in ["NPI", "TAI", "EFS", "CRI"]:
        path = PROCESSED_DIR / f"{dim}_raw.parquet"
        if path.exists():
            raw[dim] = pd.read_parquet(path)
        else:
            logger.warning("%s: no raw parquet found at %s - run data_loading.py first", dim, path)
            raw[dim] = pd.DataFrame(columns=["dimension", "task", "id", "query", "context", "ground_truth", "answer_format"])

    processed = preprocess_all(raw)
    print("\n=== Preprocessing summary ===")
    for dim, df in processed.items():
        print(f"{dim}: {len(df)} instances ready for model querying")
        if len(df):
            print("  sample zero-shot prompt:\n  " + df.iloc[0]["prompt_zero_shot"][:300].replace("\n", "\n  "))

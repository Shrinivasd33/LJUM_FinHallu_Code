"""
Phase 1 (part 1): Data loading for the Fin-Hallu Score pipeline.

Loads FinBen task datasets (per config/task_config.py) and Financial
PhraseBank, returning everything as pandas DataFrames with a consistent
column schema: [dimension, task, id, query, context, ground_truth, answer_format].

No API keys required for any of this - only an optional HF_TOKEN to unlock
the three gated FinBen datasets (see config/task_config.py docstring).
"""
import io
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.task_config import (
    FIN_HALLU_TASKS, PHRASEBANK_ZIP_URL, PHRASEBANK_AGREEMENT_FILE, PHRASEBANK_ANSWER_FORMAT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / "config" / ".env")
HF_TOKEN = os.getenv("HF_TOKEN") or None


def load_financial_phrasebank(agreement_file: str = PHRASEBANK_AGREEMENT_FILE) -> pd.DataFrame:
    """Download and parse Financial PhraseBank directly (bypasses the deprecated
    HF loading-script mechanism that the modern `datasets` library no longer supports).
    """
    cache_path = DATA_RAW / "FinancialPhraseBank-v1.0.zip"
    if cache_path.exists():
        raw_bytes = cache_path.read_bytes()
    else:
        logger.info("Downloading Financial PhraseBank...")
        resp = requests.get(PHRASEBANK_ZIP_URL, timeout=60)
        resp.raise_for_status()
        raw_bytes = resp.content
        cache_path.write_bytes(raw_bytes)

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        member = f"FinancialPhraseBank-v1.0/{agreement_file}"
        text = z.read(member).decode("latin-1")

    rows = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({
            "dimension": "EFS",
            "task": "PhraseBank",
            "id": f"phrasebank_{i}",
            "query": sentence.strip(),
            "context": None,
            "ground_truth": label.strip(),
            "answer_format": PHRASEBANK_ANSWER_FORMAT,
        })
    df = pd.DataFrame(rows)
    logger.info("Financial PhraseBank loaded: %d sentences", len(df))
    return df


def load_finben_task(dimension: str, task_cfg: dict) -> pd.DataFrame | None:
    """Load a single FinBen task dataset. Returns None (with a warning) if the
    dataset is gated and no valid HF_TOKEN is available - callers should skip
    gracefully rather than crash, so the pipeline still runs on open tasks.
    """
    repo = task_cfg["repo"]
    try:
        kwargs = {}
        if task_cfg["gated"]:
            if not HF_TOKEN:
                logger.warning(
                    "Skipping gated dataset %s (%s) - no HF_TOKEN set. "
                    "See config/task_config.py docstring to unlock it.",
                    task_cfg["name"], repo,
                )
                return None
            kwargs["token"] = HF_TOKEN
        ds = load_dataset(repo, **kwargs)
        split = task_cfg["split"] if task_cfg["split"] in ds else list(ds.keys())[0]
        table = ds[split]
        query_col = task_cfg["query_col"]
        answer_col = task_cfg["answer_col"]
        context_col = task_cfg.get("context_col")

        answer_format = task_cfg["answer_format"]
        rows = []
        for i, ex in enumerate(table):
            rows.append({
                "dimension": dimension,
                "task": task_cfg["name"],
                "id": f"{task_cfg['name'].lower()}_{i}",
                "query": ex.get(query_col),
                "context": ex.get(context_col) if context_col else None,
                "ground_truth": ex.get(answer_col),
                "answer_format": answer_format,
            })
        df = pd.DataFrame(rows)
        logger.info("%s (%s) loaded: %d examples", task_cfg["name"], repo, len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load %s (%s): %s: %s", task_cfg["name"], repo, type(e).__name__, e)
        return None


def load_all_tasks() -> dict[str, pd.DataFrame]:
    """Load every configured task. Returns {dimension: concatenated DataFrame}.
    EFS additionally includes Financial PhraseBank.
    """
    results: dict[str, list[pd.DataFrame]] = {dim: [] for dim in FIN_HALLU_TASKS}

    for dimension, tasks in FIN_HALLU_TASKS.items():
        for task_cfg in tasks:
            df = load_finben_task(dimension, task_cfg)
            if df is not None and len(df) > 0:
                results[dimension].append(df)

    results["EFS"].append(load_financial_phrasebank())

    combined = {}
    for dimension, dfs in results.items():
        if dfs:
            combined[dimension] = pd.concat(dfs, ignore_index=True)
        else:
            logger.warning("No data loaded at all for dimension %s!", dimension)
            combined[dimension] = pd.DataFrame(
                columns=["dimension", "task", "id", "query", "context", "ground_truth", "answer_format"]
            )
    return combined


if __name__ == "__main__":
    data = load_all_tasks()
    print("\n=== Load summary ===")
    for dim, df in data.items():
        print(f"{dim}: {len(df)} total examples across tasks {sorted(df['task'].unique().tolist()) if len(df) else '[]'}")
        out_path = DATA_RAW.parent / "processed" / f"{dim}_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"  saved -> {out_path}")

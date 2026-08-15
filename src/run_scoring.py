"""
Phase 4 + Phase 5 orchestration: reads every results/*.jsonl file produced by
run_experiments.py, scores it with fin_hallu_score.py, runs the significance
tests in stats_analysis.py, and builds the Strategic Deployment Matrix.

No API keys, no cost, no network access - pure post-processing of whatever
has already been collected. Safe to re-run any time (e.g. after each new
batch of experiments finishes) to refresh the analysis with partial data.

Usage:
    python src/run_scoring.py
"""
import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fin_hallu_score import score_dataframe, compute_fin_hallu_score
from src.stats_analysis import compare_conditions, compare_models_anova
from src.deployment_matrix import build_deployment_matrix, summarize_for_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_all_results() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        try:
            df = pd.read_json(path, lines=True)
            if not df.empty:
                frames.append(df)
        except ValueError:
            logger.warning("Could not parse %s - skipping", path)
    if not frames:
        return pd.DataFrame(
            columns=["dimension", "task", "id", "model", "condition", "run_idx", "ground_truth", "model_output", "answer_format"]
        )
    return pd.concat(frames, ignore_index=True)


def main():
    raw = load_all_results()
    if raw.empty:
        logger.error(
            "No results found in %s - run src/run_experiments.py first "
            "(needs API keys in config/.env).", RESULTS_DIR
        )
        return

    logger.info("Loaded %d raw results across dimensions %s", len(raw), sorted(raw["dimension"].unique()))

    scored = score_dataframe(raw)
    scored.to_csv(RESULTS_DIR / "scored_instances.csv", index=False)
    logger.info("Saved per-instance scores -> %s", RESULTS_DIR / "scored_instances.csv")

    # --- Aggregate to (dimension, model, condition) mean F1 ---
    summary = summarize_for_matrix(scored)
    summary.to_csv(RESULTS_DIR / "index_score_summary.csv", index=False)
    print("\n=== Fin-Hallu index scores by dimension x model x condition ===")
    print(summary.to_string(index=False))

    # --- Composite Fin-Hallu Score per (model, condition), averaging the 4 indices ---
    pivot = summary.pivot_table(index=["model", "condition"], columns="dimension", values="index_score")
    for dim in ["NPI", "TAI", "EFS", "CRI"]:
        if dim not in pivot.columns:
            pivot[dim] = float("nan")
    pivot["Fin-Hallu Score"] = pivot.apply(
        lambda r: compute_fin_hallu_score(r["NPI"], r["TAI"], r["EFS"], r["CRI"]), axis=1
    )
    pivot.to_csv(RESULTS_DIR / "composite_fin_hallu_scores.csv")
    print("\n=== Composite Fin-Hallu Score (equal weights) ===")
    print(pivot.to_string())

    # --- Strategic Deployment Matrix ---
    matrix = build_deployment_matrix(summary)
    matrix.to_csv(RESULTS_DIR / "strategic_deployment_matrix.csv")
    print("\n=== Strategic Deployment Matrix ===")
    print(matrix.to_string())

    # --- RQ2/RQ3 significance tests: zero-shot vs RAG, per (dimension, model) ---
    print("\n=== RQ2: zero-shot vs RAG significance (paired per-instance F1) ===")
    for (dimension, model), group in scored.groupby(["dimension", "model"]):
        # Pair by (id, run_idx), not just id - a partially-collected condition
        # (e.g. a model still mid-RAG-collection) can have fewer repeat runs
        # for a given id than its zero-shot counterpart already has, and
        # id-only pairing silently mismatches lengths in that case.
        zs = group[group["condition"] == "zero_shot"].copy()
        rag = group[group["condition"] == "rag"].copy()
        zs["key"] = list(zip(zs["id"], zs["run_idx"]))
        rag["key"] = list(zip(rag["id"], rag["run_idx"]))
        common_keys = set(zs["key"]) & set(rag["key"])
        if len(common_keys) < 2:
            continue
        zs_f1 = zs[zs["key"].isin(common_keys)].sort_values("key")["f1"].tolist()
        rag_f1 = rag[rag["key"].isin(common_keys)].sort_values("key")["f1"].tolist()
        result = compare_conditions(rag_f1, zs_f1)  # positive mean_diff = RAG helped
        print(f"  [n={len(common_keys)} paired instances]", end=" ")
        direction = "RAG HELPED" if result.mean_diff > 0 else "RAG HURT" if result.mean_diff < 0 else "no change"
        print(
            f"{dimension} / {model}: {direction} (mean diff={result.mean_diff:+.3f}, "
            f"{result.test_used}, p={result.p_value:.4f}, significant={result.significant})"
        )

    # --- ANOVA across models per dimension ---
    print("\n=== ANOVA: differences across LLM architectures, per dimension ===")
    for dimension, group in scored.groupby("dimension"):
        by_model = {m: g["f1"].tolist() for m, g in group.groupby("model") if len(g) >= 2}
        if len(by_model) < 2:
            continue
        result = compare_models_anova(by_model)
        print(f"{dimension}: F={result['f_statistic']:.3f}, p={result['p_value']:.4f}, significant={result['significant']}")


if __name__ == "__main__":
    main()

"""
Phase 5 (part 2): Strategic Deployment Matrix construction, per Chapter 3,
Section 3.5 (Table 3.4's illustrative structure) - the second primary output
of the thesis alongside the Fin-Hallu Score itself.

Takes scored results (one row per model x condition x dimension, with the
aggregated index score for that combination) and picks, for each
(dimension, model) pair, which condition (zero_shot / few_shot / rag)
produced the lowest hallucination (= highest F1/accuracy), labelling cells
as Baseline / Preferred / Test / Caution per the proposal's matrix legend.
"""
import pandas as pd


def build_deployment_matrix(scored_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Input: scored_summary must have columns [dimension, model, condition, index_score]
           - one row per (dimension, model, condition) with its mean F1/accuracy.
    Output: a matrix with rows = dimensions, columns = model, cells = recommendation
            string like "Preferred: few_shot (F1=0.82)".
    """
    required = {"dimension", "model", "condition", "index_score"}
    missing = required - set(scored_summary.columns)
    if missing:
        raise ValueError(f"scored_summary is missing columns: {missing}")

    rows = []
    for dimension, dim_group in scored_summary.groupby("dimension"):
        row = {"dimension": dimension}
        for model, model_group in dim_group.groupby("model"):
            best = model_group.loc[model_group["index_score"].idxmax()]
            baseline = model_group[model_group["condition"] == "zero_shot"]
            baseline_score = baseline["index_score"].iloc[0] if len(baseline) else None

            if best["condition"] == "zero_shot":
                label = f"Baseline (zero-shot): F1={best['index_score']:.2f}"
            else:
                improvement = (
                    f" (+{best['index_score'] - baseline_score:.2f} vs zero-shot)"
                    if baseline_score is not None else ""
                )
                label = f"Preferred: {best['condition']}: F1={best['index_score']:.2f}{improvement}"

            row[model] = label
        rows.append(row)

    return pd.DataFrame(rows).set_index("dimension")


def summarize_for_matrix(raw_scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse a full per-instance scored DataFrame (with columns
    dimension, model, condition, f1) down to the per-(dimension, model,
    condition) mean index score that build_deployment_matrix expects."""
    return (
        raw_scored.groupby(["dimension", "model", "condition"])["f1"]
        .mean()
        .reset_index()
        .rename(columns={"f1": "index_score"})
    )

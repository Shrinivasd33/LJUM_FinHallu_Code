"""
Phase 5 (part 1): Statistical significance testing, per Chapter 3, Section 3.7:
  - paired t-test + confidence intervals between mitigation strategies
  - Wilcoxon signed-rank test where score distributions are non-normal
  - ANOVA for differences in Fin-Hallu Scores across the three LLM architectures
  - significance threshold p < 0.05 throughout
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    test_used: str
    statistic: float
    p_value: float
    significant: bool
    mean_diff: float
    ci_95: tuple[float, float]


def _is_normal(sample: np.ndarray, alpha: float = 0.05) -> bool:
    """Shapiro-Wilk normality check to decide t-test vs Wilcoxon.
    Falls back to 'assume non-normal' for very small samples where
    Shapiro-Wilk is unreliable (n < 8)."""
    if len(sample) < 8:
        return False
    _, p = stats.shapiro(sample)
    return p > alpha


def compare_conditions(scores_a: list[float], scores_b: list[float], alpha: float = 0.05) -> ComparisonResult:
    """Compare paired per-instance F1 scores between two conditions (e.g.
    zero-shot vs RAG) for the same model/task. `scores_a` and `scores_b` must
    be the same length and instance-aligned."""
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must be the same length (paired data)")

    diffs = a - b
    mean_diff = float(np.mean(diffs))
    se = stats.sem(diffs) if len(diffs) > 1 else 0.0
    ci = stats.t.interval(1 - alpha, len(diffs) - 1, loc=mean_diff, scale=se) if len(diffs) > 1 and se > 0 else (mean_diff, mean_diff)

    if np.allclose(diffs, 0):
        # No variability at all between the two conditions - t-test/Wilcoxon
        # are mathematically undefined (0/0) here; trivially "no difference".
        return ComparisonResult(
            test_used="none (identical paired scores)", statistic=0.0, p_value=1.0,
            significant=False, mean_diff=0.0, ci_95=(0.0, 0.0),
        )

    if _is_normal(diffs, alpha):
        stat, p = stats.ttest_rel(a, b)
        test_used = "paired t-test"
    else:
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            # all-zero differences - Wilcoxon undefined, treat as no difference
            stat, p = 0.0, 1.0
        test_used = "Wilcoxon signed-rank"

    return ComparisonResult(
        test_used=test_used, statistic=float(stat), p_value=float(p),
        significant=p < alpha, mean_diff=mean_diff, ci_95=(float(ci[0]), float(ci[1])),
    )


def compare_models_anova(scores_by_model: dict[str, list[float]]) -> dict:
    """One-way ANOVA across the three LLM architectures' Fin-Hallu Scores."""
    groups = list(scores_by_model.values())
    stat, p = stats.f_oneway(*groups)
    return {"f_statistic": float(stat), "p_value": float(p), "significant": p < 0.05}

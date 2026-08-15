"""
Tests for Phase 5 modules (stats_analysis, deployment_matrix).
Run with: pytest tests/test_phase5.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.stats_analysis import compare_conditions, compare_models_anova
from src.deployment_matrix import build_deployment_matrix, summarize_for_matrix


class TestStatsAnalysis:
    def test_identical_conditions_not_significant(self):
        rng = np.random.default_rng(0)
        a = list(rng.normal(0.7, 0.05, 30))
        result = compare_conditions(a, a)
        assert result.p_value > 0.05
        assert bool(result.significant) is False
        assert result.mean_diff == pytest.approx(0.0)

    def test_clearly_different_conditions_significant(self):
        rng = np.random.default_rng(1)
        a = list(rng.normal(0.9, 0.05, 30))
        b = list(rng.normal(0.3, 0.05, 30))
        result = compare_conditions(a, b)
        assert bool(result.significant) is True
        assert result.mean_diff > 0.5

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            compare_conditions([1, 2, 3], [1, 2])

    def test_anova_detects_model_difference(self):
        rng = np.random.default_rng(2)
        scores = {
            "gpt-4": list(rng.normal(0.9, 0.03, 20)),
            "llama-3-70b": list(rng.normal(0.6, 0.03, 20)),
            "gemini-1.5": list(rng.normal(0.3, 0.03, 20)),
        }
        result = compare_models_anova(scores)
        assert bool(result["significant"]) is True


class TestDeploymentMatrix:
    def test_summarize_and_build(self):
        raw = pd.DataFrame({
            "dimension": ["NPI"] * 6,
            "model": ["gpt-4"] * 3 + ["llama-3-70b"] * 3,
            "condition": ["zero_shot", "few_shot", "rag"] * 2,
            "f1": [0.6, 0.8, 0.7, 0.4, 0.5, 0.9],
        })
        summary = summarize_for_matrix(raw)
        assert len(summary) == 6

        matrix = build_deployment_matrix(summary)
        assert "gpt-4" in matrix.columns
        assert "llama-3-70b" in matrix.columns
        assert "NPI" in matrix.index
        # gpt-4's best condition should be few_shot (0.8)
        assert "few_shot" in matrix.loc["NPI", "gpt-4"]
        # llama-3-70b's best condition should be rag (0.9)
        assert "rag" in matrix.loc["NPI", "llama-3-70b"]

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"dimension": ["NPI"], "model": ["gpt-4"]})
        with pytest.raises(ValueError):
            build_deployment_matrix(bad_df)

    def test_zero_shot_best_labelled_baseline_not_preferred(self):
        raw = pd.DataFrame({
            "dimension": ["CRI"] * 3,
            "model": ["gpt-4"] * 3,
            "condition": ["zero_shot", "few_shot", "rag"],
            "f1": [0.9, 0.5, 0.4],  # zero-shot already best - RAG/few-shot hurt
        })
        summary = summarize_for_matrix(raw)
        matrix = build_deployment_matrix(summary)
        assert "Baseline" in matrix.loc["CRI", "gpt-4"]

"""
Unit tests for the Fin-Hallu Score module. Run with:
    pytest tests/test_fin_hallu_score.py -v
No API keys, no network access, no cost - pure logic tests with synthetic data.

Scoring is dispatched by `answer_format`, not by Fin-Hallu dimension (dimensions
are no longer format-homogeneous - see fin_hallu_score.py's module docstring).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.fin_hallu_score import (
    score_instance, score_dataframe, aggregate_index_score,
    compute_fin_hallu_score, cri_inter_rater_kappa,
)


class TestNumeric:
    def test_exact_number_match(self):
        r = score_instance("numeric", "193 million", "193")
        assert r.f1 == pytest.approx(1.0)
        assert r.parsed

    def test_currency_and_comma_formatting_ignored(self):
        r = score_instance("numeric", "$1,234.50", "1234.5")
        assert r.f1 == pytest.approx(1.0)

    def test_wrong_number(self):
        r = score_instance("numeric", "500", "193")
        assert r.f1 == 0.0

    def test_null_output_counts_as_hallucination(self):
        r = score_instance("numeric", None, "193")
        assert r.parsed is False
        assert r.f1 == 0.0

    def test_partial_match_multi_number(self):
        # predicted has one right, one wrong extra number; truth has two numbers
        r = score_instance("numeric", "193 and 500", "193 and 250")
        assert 0.0 < r.f1 < 1.0


class TestSingleLabel:
    def test_exact_label_match(self):
        r = score_instance("single_label", "positive", "positive")
        assert r.f1 == pytest.approx(1.0)

    def test_case_insensitive(self):
        r = score_instance("single_label", "HAWKISH", "hawkish")
        assert r.f1 == pytest.approx(1.0)

    def test_wrong_label(self):
        r = score_instance("single_label", "negative", "positive")
        assert r.f1 == 0.0

    def test_null_output_counts_as_hallucination(self):
        r = score_instance("single_label", None, "dovish")
        assert r.parsed is False
        assert r.f1 == 0.0


class TestBioSequence:
    def test_exact_sequence_match(self):
        pred = "Apple:B-ORG\nreported:O\nrevenue:O"
        truth = "Apple:B-ORG\nreported:O\nrevenue:O"
        r = score_instance("bio_sequence", pred, truth)
        assert r.f1 == pytest.approx(1.0)

    def test_all_O_ground_truth_and_prediction(self):
        # No entities anywhere - nothing to match on, treated as no signal
        # (empty truth set) rather than a false "perfect" score.
        pred = "The:O\ncat:O\nsat:O"
        truth = "The:O\ncat:O\nsat:O"
        r = score_instance("bio_sequence", pred, truth)
        assert r.f1 == 0.0

    def test_partial_span_overlap(self):
        pred = "Apple:B-ORG\nInc:I-ORG\nreported:O"
        truth = "Apple:B-ORG\nInc:I-ORG\nrevenue:B-NetIncomeLoss"
        r = score_instance("bio_sequence", pred, truth)
        assert 0.0 < r.f1 < 1.0

    def test_missed_entity(self):
        pred = "The:O\ncat:O\nsat:O"
        truth = "The:O\ncat:B-PER\nsat:O"
        r = score_instance("bio_sequence", pred, truth)
        assert r.f1 == 0.0

    def test_case_and_whitespace_normalised(self):
        pred = " Apple : b-org \n reported : o "
        truth = "Apple:B-ORG\nreported:O"
        r = score_instance("bio_sequence", pred, truth)
        assert r.f1 == pytest.approx(1.0)


class TestEntityList:
    def test_exact_match(self):
        r = score_instance("entity_list", "Apple Inc, ORG\nJohn Smith, PER", "Apple Inc, ORG\nJohn Smith, PER")
        assert r.f1 == pytest.approx(1.0)

    def test_partial_overlap(self):
        r = score_instance("entity_list", "Apple Inc, ORG", "Apple Inc, ORG\nJohn Smith, PER")
        assert 0.0 < r.f1 < 1.0

    def test_wrong_type_counts_as_miss(self):
        r = score_instance("entity_list", "Apple Inc, PER", "Apple Inc, ORG")
        assert r.f1 == 0.0


class TestTripletList:
    def test_exact_match(self):
        pred = "Apple ; Tim Cook ; ceo_of"
        truth = "Apple ; Tim Cook ; ceo_of"
        r = score_instance("triplet_list", pred, truth)
        assert r.f1 == pytest.approx(1.0)

    def test_wrong_relation(self):
        r = score_instance("triplet_list", "Apple ; Tim Cook ; founder_of", "Apple ; Tim Cook ; ceo_of")
        assert r.f1 == 0.0

    def test_multi_triplet_partial_overlap(self):
        pred = "Apple ; Tim Cook ; ceo_of"
        truth = "Apple ; Tim Cook ; ceo_of\nApple ; Cupertino ; headquartered_in"
        r = score_instance("triplet_list", pred, truth)
        assert 0.0 < r.f1 < 1.0

    def test_malformed_line_ignored(self):
        # missing one ';' separator - should not crash, just contributes nothing
        r = score_instance("triplet_list", "Apple Tim Cook ceo_of", "Apple ; Tim Cook ; ceo_of")
        assert r.f1 == 0.0


class TestNumericOrText:
    def test_falls_back_to_numeric_when_truth_has_numbers(self):
        r = score_instance("numeric_or_text", "42", "42")
        assert r.f1 == pytest.approx(1.0)

    def test_falls_back_to_word_overlap_when_truth_is_text(self):
        r = score_instance("numeric_or_text", "increase", "increase")
        assert r.f1 == pytest.approx(1.0)


class TestUnknownFormat:
    def test_raises_on_unknown_answer_format(self):
        with pytest.raises(ValueError):
            score_instance("not_a_real_format", "x", "y")


class TestAggregation:
    def test_score_dataframe_and_aggregate(self):
        df = pd.DataFrame({
            "model_output": ["193", "500", None],
            "ground_truth": ["193", "193", "193"],
            "answer_format": ["numeric", "numeric", "numeric"],
        })
        scored = score_dataframe(df)
        assert scored["parsed"].tolist() == [True, True, False]
        # one perfect (1.0), one wrong (0.0), one null (0.0) -> mean F1 = 1/3
        assert aggregate_index_score(scored) == pytest.approx(1 / 3)

    def test_score_dataframe_handles_mixed_answer_formats(self):
        # Different tasks within the same dimension can use different formats -
        # this is the whole point of the per-row answer_format redesign.
        df = pd.DataFrame({
            "model_output": ["193", "positive", "Apple, ORG"],
            "ground_truth": ["193", "positive", "Apple, ORG"],
            "answer_format": ["numeric", "single_label", "entity_list"],
        })
        scored = score_dataframe(df)
        assert scored["f1"].tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_composite_score_equal_weights(self):
        score = compute_fin_hallu_score(npi=1.0, tai=1.0, efs=1.0, cri=1.0)
        assert score == pytest.approx(1.0)
        score = compute_fin_hallu_score(npi=0.0, tai=0.0, efs=0.0, cri=0.0)
        assert score == pytest.approx(0.0)

    def test_composite_score_weighted_variant_favours_npi_cri(self):
        # NPI/CRI perfect, TAI/EFS zero - weighted variant should score higher
        # than equal-weight variant, since NPI+CRI get 70% combined instead of 50%.
        equal = compute_fin_hallu_score(npi=1.0, tai=0.0, efs=0.0, cri=1.0, weighted=False)
        weighted = compute_fin_hallu_score(npi=1.0, tai=0.0, efs=0.0, cri=1.0, weighted=True)
        assert weighted > equal
        assert weighted == pytest.approx(0.70)
        assert equal == pytest.approx(0.50)


class TestKappa:
    def test_perfect_agreement(self):
        k = cri_inter_rater_kappa([1, 0, 1, 1, 0], [1, 0, 1, 1, 0])
        assert k == pytest.approx(1.0)

    def test_no_agreement_beyond_chance(self):
        # Deliberately anti-correlated - kappa should be low/negative
        k = cri_inter_rater_kappa([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1])
        assert k < 0.5

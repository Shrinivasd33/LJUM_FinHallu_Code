"""
Phase 4: Fin-Hallu Score computation - the thesis's core methodological
contribution (Chapter 3, Section 3.5 and 3.8).

Fin-Hallu Score = w1*NPI + w2*TAI + w3*EFS + w4*CRI  (each index in [0, 1])

Design note (documented, not hidden): FinBen's real task ground-truth
formats are heterogeneous - inspecting them directly (2026-07-19) showed
several are full per-token BIO sequence-tagging outputs (FNXL, FiNER-Ord,
CausalDetection), not short answers. Rather than four bespoke scoring
functions per *dimension*, this module dispatches on each task's
`answer_format` (set in config/task_config.py) using ONE general set-overlap
precision/recall/F1 engine with a format-specific *normalizer* that extracts
the right kind of "item" from free text:
  - "numeric"      : numeric tokens (FinQA)
  - "single_label" : whole answer as one label (FOMC, PhraseBank) - P=R=F1
                     collapses to plain accuracy, matching Chapter 3.8's
                     "correct / total" formula for Temporal Accuracy /
                     Logical Consistency
  - "bio_sequence" : (token, tag) pairs, entity tags only (FNXL, FiNER-Ord,
                     CausalDetection) - standard entity-F1, same convention
                     as CoNLL-style NER evaluation
  - "entity_list"  : (name, type) pairs (NER)
  - "triplet_list" : (head, tail, relation) triples (FinRED)
  - "numeric_or_text": numeric-set F1 if ground truth contains numbers,
                     else word-overlap F1 (kept for completeness; not
                     currently used by any active task - see task_config.py's
                     note on why TAT-QA is excluded)
"""
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _normalize_numbers(text) -> set[str]:
    """Extract numeric values from text, normalising $/,/% formatting so
    '$1,234.5 million' and '1234.5' are comparable."""
    if text is None:
        return set()
    text = str(text)
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    out = set()
    for m in matches:
        cleaned = m.replace(",", "")
        try:
            val = float(cleaned)
            out.add(f"{val:.2f}")
        except ValueError:
            continue
    return out


def _normalize_single_label(text) -> set[str]:
    """Whole answer is one label - normalize case/whitespace/punctuation."""
    if text is None:
        return set()
    cleaned = re.sub(r"[^\w\s-]", "", str(text)).strip().lower()
    return {cleaned} if cleaned else set()


def _normalize_bio_sequence(text) -> set[str]:
    """Parse 'token:TAG' per-line output into a set of (token, tag) pairs,
    keeping only entity tags (dropping 'O') so scoring focuses on whether
    entities/spans were correctly identified - the standard convention for
    sequence-labelling F1 (majority-class 'O' tokens would trivially inflate
    a naive per-token accuracy otherwise)."""
    if text is None:
        return set()
    pairs = set()
    for line in str(text).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        token, tag = line.rsplit(":", 1)
        token, tag = token.strip().lower(), tag.strip().upper()
        if tag and tag != "O":
            pairs.add(f"{token}::{tag}")
    return pairs


def _normalize_entity_list(text) -> set[str]:
    """Parse 'Name, TYPE' per-line output into a set of (name, type) pairs."""
    if text is None:
        return set()
    pairs = set()
    for line in str(text).splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        name, etype = line.rsplit(",", 1)
        pairs.add(f"{name.strip().lower()}::{etype.strip().upper()}")
    return pairs


def _normalize_triplet_list(text) -> set[str]:
    """Parse 'head ; tail ; relation' per-line output into a set of triples."""
    if text is None:
        return set()
    triples = set()
    for line in str(text).splitlines():
        parts = [p.strip().lower() for p in line.split(";")]
        if len(parts) == 3 and all(parts):
            triples.add("::".join(parts))
    return triples


def _normalize_numeric_or_text(text) -> set[str]:
    """Fallback for mixed numeric/text ground truth: number-set if any
    numbers are present, else lowercase word-overlap set."""
    nums = _normalize_numbers(text)
    if nums:
        return nums
    if text is None:
        return set()
    words = re.findall(r"\b\w+\b", str(text).lower())
    return set(words)


NORMALIZERS = {
    "numeric": _normalize_numbers,
    "single_label": _normalize_single_label,
    "bio_sequence": _normalize_bio_sequence,
    "entity_list": _normalize_entity_list,
    "triplet_list": _normalize_triplet_list,
    "numeric_or_text": _normalize_numeric_or_text,
}

# Which Fin-Hallu dimension each answer_format most commonly maps into is
# already handled upstream by config/task_config.py's FIN_HALLU_TASKS
# structure - this module only needs the format, not the dimension.


@dataclass
class ScoreResult:
    precision: float
    recall: float
    f1: float
    parsed: bool  # False if model_output was null/unparseable (counts as a hallucination error)


def score_instance(answer_format: str, model_output, ground_truth) -> ScoreResult:
    if answer_format not in NORMALIZERS:
        raise ValueError(f"Unknown answer_format '{answer_format}'. Choose from {list(NORMALIZERS)}")
    normalizer = NORMALIZERS[answer_format]

    if model_output is None or (isinstance(model_output, float) and pd.isna(model_output)):
        return ScoreResult(precision=0.0, recall=0.0, f1=0.0, parsed=False)

    pred_set = normalizer(model_output)
    truth_set = normalizer(ground_truth)

    if not truth_set:
        return ScoreResult(precision=0.0, recall=0.0, f1=0.0, parsed=len(pred_set) > 0)

    if not pred_set:
        return ScoreResult(precision=0.0, recall=0.0, f1=0.0, parsed=False)

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ScoreResult(precision=precision, recall=recall, f1=f1, parsed=True)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: model_output, ground_truth, answer_format.
    Adds precision/recall/f1/parsed. (Previously this took a `dimension`
    argument and used one normalizer per dimension - now driven by the
    per-row `answer_format` column instead, since dimensions are no longer
    format-homogeneous.)"""
    results = df.apply(
        lambda row: score_instance(row["answer_format"], row["model_output"], row["ground_truth"]), axis=1
    )
    df = df.copy()
    df["precision"] = [r.precision for r in results]
    df["recall"] = [r.recall for r in results]
    df["f1"] = [r.f1 for r in results]
    df["parsed"] = [r.parsed for r in results]
    return df


def aggregate_index_score(scored_df: pd.DataFrame) -> float:
    """The dimension-level index value (0-1) = mean F1 across all scored instances."""
    if scored_df.empty:
        return float("nan")
    return scored_df["f1"].mean()


def compute_fin_hallu_score(
    npi: float, tai: float, efs: float, cri: float, weighted: bool = False
) -> float:
    """Chapter 3.5: equal weights (0.25 each) by default; optional weighted
    variant gives 35% to NPI+CRI (high-stakes, per RQ3) and 15% to TAI+EFS."""
    if weighted:
        w = {"npi": 0.35, "tai": 0.15, "efs": 0.15, "cri": 0.35}
    else:
        w = {"npi": 0.25, "tai": 0.25, "efs": 0.25, "cri": 0.25}
    return w["npi"] * npi + w["tai"] * tai + w["efs"] * efs + w["cri"] * cri


def cri_inter_rater_kappa(rater1_labels: list, rater2_labels: list) -> float:
    """Cohen's Kappa for CRI inter-rater agreement (Chapter 3.5/3.8). Only
    meaningful if you have two independent judgments per causal claim - e.g.
    two human reviewers, or a human + an LLM-as-judge second pass."""
    return cohen_kappa_score(rater1_labels, rater2_labels)

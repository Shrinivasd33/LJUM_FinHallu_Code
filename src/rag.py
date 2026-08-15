"""
Phase 3: Retrieval-Augmented Generation (RAG) module, per Chapter 3, Section
3.7 ("retrieved context from financial reference sources... is added").

Design choice (documented honestly, not hidden): rather than requiring a
separately-licensed external corpus of annual reports, this builds the
retrieval corpus from the `context` passages already present across each
FinBen task's sampled instances (excluding the query's own context), plus a
small hand-written financial glossary/fiscal-calendar reference supplement.
This keeps the pipeline runnable entirely on free-tier resources with only
`scikit-learn` (already installed) and no heavyweight embedding models,
while still genuinely testing RQ2 (does added context help or hurt).

Uses TF-IDF + cosine similarity - simple, fast, dependency-light, and a
completely standard baseline retriever choice for this kind of ablation.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent

# Small hand-authored reference supplement - genuinely external knowledge,
# not derived from any task's own context, so it adds real information
# rather than just re-surfacing another sample from the same benchmark.
FINANCIAL_GLOSSARY = [
    "A fiscal quarter (Q1-Q4) does not always align with the calendar quarter; "
    "many US companies use a fiscal year ending in a month other than December.",
    "FOMC (Federal Open Market Committee) meets eight times per year to set US "
    "monetary policy; a 'hawkish' stance favours tightening (raising rates) to "
    "curb inflation, a 'dovish' stance favours easing (cutting rates) to support growth.",
    "EPS (Earnings Per Share) = Net Income / Weighted Average Shares Outstanding.",
    "A 10-K is an annual report filed with the US SEC; a 10-Q is the quarterly "
    "equivalent, filed within 40-45 days of quarter-end.",
    "Named entity types in financial NER commonly include: PERSON (executives, "
    "analysts), ORGANIZATION (companies, regulators), LOCATION (markets, "
    "jurisdictions), and MONEY/PERCENT (financial figures).",
    "A causal claim in financial text asserts that one event (e.g. a rate "
    "decision) produced another (e.g. a price movement); correlation alone "
    "(two events co-occurring) does not establish causation.",
]


class SimpleRetriever:
    def __init__(self, corpus_passages: list[str]):
        self.passages = corpus_passages
        if corpus_passages:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            self.matrix = self.vectorizer.fit_transform(corpus_passages)
        else:
            self.vectorizer = None
            self.matrix = None

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        if not self.passages or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [self.passages[i] for i in top_idx if sims[i] > 0]


def build_retriever_for_task(df: pd.DataFrame, task_name: str, exclude_id: str | None = None) -> SimpleRetriever:
    """Build a retriever over all context passages for one task (excluding the
    current instance, if given, so a query never retrieves its own context)."""
    task_rows = df[df["task"] == task_name]
    if exclude_id is not None:
        task_rows = task_rows[task_rows["id"] != exclude_id]
    passages = [c for c in task_rows["context"].dropna().unique().tolist() if str(c).strip()]
    passages = list(dict.fromkeys(passages))  # de-dup, preserve order
    passages.extend(FINANCIAL_GLOSSARY)
    return SimpleRetriever(passages)


def augment_prompt(base_prompt: str, retrieved: list[str]) -> str:
    """Prepend retrieved passages to an existing zero-shot/few-shot prompt,
    clearly labelled as retrieved (not the primary context) so the model can
    distinguish it - and so we can later test whether this framing itself
    affects the RAG crossover point (RQ2)."""
    if not retrieved:
        return base_prompt
    block = "\n\n".join(f"[Retrieved reference {i+1}]: {p}" for i, p in enumerate(retrieved))
    return f"Additional retrieved context that may or may not be relevant:\n{block}\n\n{base_prompt}"


if __name__ == "__main__":
    # Self-contained smoke test with synthetic data - no API calls, no cost.
    sample_df = pd.DataFrame({
        "task": ["FOMC"] * 3,
        "id": ["a", "b", "c"],
        "context": [
            "The committee decided to raise the federal funds rate by 25 basis points to combat inflation.",
            "Officials signalled a pause in rate hikes given softening labour market data.",
            "The Fed left rates unchanged, citing balanced risks to its dual mandate.",
        ],
    })
    retriever = build_retriever_for_task(sample_df, "FOMC", exclude_id="a")
    results = retriever.retrieve("Is the Fed raising or cutting rates?", k=2)
    print("Retrieved passages:")
    for r in results:
        print(" -", r[:100])
    print("\nAugmented prompt:\n", augment_prompt("Classify the stance.", results))

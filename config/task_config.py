"""
Task configuration for the Fin-Hallu Score experiment pipeline.

Maps each of the four Fin-Hallu dimensions (NPI, TAI, EFS, CRI) to the real,
verified FinBen (TheFinAI/PIXIU) HuggingFace dataset repos that supply it,
per Chapter 3, Section 3.3 (Dataset Description) of the thesis.

IMPORTANT - per-task `answer_format`, not one instruction per dimension:
Inspecting the real ground-truth schemas on 2026-07-19 revealed FinBen tasks
are far more heterogeneous than "short answer" - several are full per-token
BIO sequence-tagging tasks (label every token), not simple QA. Using one
instruction/scorer per *dimension* would have silently produced meaningless
results (e.g. asking a model to "reply with the number only" on a task whose
ground truth is a 40-token BIO tag sequence). Each task below therefore
carries its own `answer_format`, which fin_hallu_score.py dispatches on:

  - "numeric"        : short numeric answer (FinQA)                         -> number-set F1
  - "numeric_or_text" : TAT-QA's mixed numeric/text/list answers            -> number-set F1 if truth has numbers, else word-overlap F1
  - "single_label"    : one classification label (FOMC, PhraseBank)         -> exact-match (collapses precision=recall=F1=accuracy)
  - "bio_sequence"    : per-token B-/I-/O tag sequence (FNXL, FiNER-Ord,
                        CausalDetection)                                    -> token-level tag F1
  - "entity_list"     : newline-separated "Name, TYPE" pairs (NER)          -> pair-set F1
  - "triplet_list"     : "head ; tail ; relation" triplets (FinRED)          -> triplet-set F1

IMPORTANT - gated datasets:
Three FinBen datasets require a one-time, free HuggingFace agreement before
they can be downloaded (this is a licensing click-through, not a paywall):
  - TheFinAI/flare-finqa        (NPI - FinQA)
  - TheFinAI/flare-cd           (CRI - Causal Detection)
  - TheFinAI/flare-ectsum       (not used - see note under TAI below)

To unlock:
  1. Create a free account at https://huggingface.co/join
  2. Visit each dataset page above and click "Agree and access repository"
  3. Generate a read token at https://huggingface.co/settings/tokens
  4. Put it in config/.env as HF_TOKEN=hf_xxxxxxxx
"""

FIN_HALLU_TASKS = {
    "NPI": [  # Numerical Precision Index
        {
            "name": "FinQA",
            "repo": "TheFinAI/flare-finqa",
            "gated": True,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "numeric",
        },
        {
            "name": "FNXL",
            "repo": "TheFinAI/flare-fnxl",
            "gated": False,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "bio_sequence",
        },
        # TAT-QA (TheFinAI/flare-tatqa) is EXCLUDED. Its ground truth mixes
        # numeric answers, date/year lists, and free-text explanatory spans
        # within the same task, with no reliable field distinguishing them
        # in advance. Rather than silently score free-text answers as
        # numeric failures (deflating NPI for reasons that have nothing to
        # do with the model's numerical accuracy), it's left out. Revisit
        # if a per-instance answer-type classifier is built later.
    ],
    "TAI": [  # Temporal Alignment Index
        {
            "name": "FOMC",
            "repo": "TheFinAI/flare-fomc",
            "gated": False,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "single_label",
        },
        # ECTSum (TheFinAI/flare-ectsum) is EXCLUDED. Its "answer" field is a
        # long newline-separated sequence of 0s/1s marking which transcript
        # sentences belong in an extractive summary - an extractive
        # summarisation task, not a date/fiscal-period label. Testing
        # sentence-selection accuracy would not measure "temporal alignment"
        # as Chapter 3 defines it. See README's "Known methodology deviations".
    ],
    "EFS": [  # Entity Fidelity Score
        {
            "name": "NER",
            "repo": "TheFinAI/flare-ner",
            "gated": False,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "entity_list",
        },
        {
            "name": "FiNER-Ord",
            "repo": "TheFinAI/flare-finer-ord",
            "gated": False,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "bio_sequence",
        },
        {
            "name": "FinRED",
            "repo": "TheFinAI/flare-finred",
            "gated": False,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "triplet_list",
        },
    ],
    "CRI": [  # Causal Reasoning Integrity
        {
            "name": "CausalDetection",
            "repo": "TheFinAI/flare-cd",
            "gated": True,
            "split": "test",
            "query_col": "query",
            "answer_col": "answer",
            "context_col": "text",
            "answer_format": "bio_sequence",
        },
        # FinCausal20-Task1 (TheFinAI/flare-causal20-sc) is a SEPARATELY
        # gated dataset (its own "Agree and access" click needed, distinct
        # from flare-cd) - not required, CausalDetection alone gives CRI
        # real data. Add it later if you want a second CRI source:
        # https://huggingface.co/datasets/TheFinAI/flare-causal20-sc
    ],
}

# NOTE - no per-task instruction dict here (removed 2026-07-19): inspecting
# FinBen's native `query` field for every task confirmed it already contains
# the complete, precise, self-sufficient instruction (exact tag vocabulary,
# format spec, and the "Text/Context: ...\nAnswer:" structure) - anything
# written here would only duplicate or risk contradicting it. PhraseBank has
# no native instruction (it's a raw sentence@label file), so preprocessing.py
# carries one small task-specific instruction for it alone.
PHRASEBANK_INSTRUCTION = (
    "Classify the sentiment of the following financial news sentence as "
    "exactly one of: positive, negative, neutral. Reply with that one word "
    "only, no explanation."
)

# Financial PhraseBank - used for EFS (entity-level sentiment), per Chapter 3.
# Not on modern `datasets` library (legacy loading script) - downloaded directly.
PHRASEBANK_ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
PHRASEBANK_AGREEMENT_FILE = "Sentences_AllAgree.txt"  # 100% agreement, 2264 sentences
PHRASEBANK_ANSWER_FORMAT = "single_label"

# Models under evaluation (Chapter 3, Table 3.3)
# NOTE 2026-07-26: registry key stays "gpt-4" (matches CLI usage and existing
# results file naming) but model_id points at gpt-4o - classic gpt-4 pricing
# at our ~4,800-call volume would exceed $100; gpt-4o keeps this to a
# reasonable ~$15-25 while remaining OpenAI's flagship GPT-4-family model.
# Document this in Chapter 3/Table 3.3 when finalising.
# NOTE 2026-07-26: Google retired gemini-1.5-pro AND gemini-2.5-flash for new
# API projects (404 "no longer available to new users") before Phase 2
# execution - the model-version-drift risk already flagged in the thesis
# limitations. Verified against the live API: gemini-3.5-flash is the current
# stable pinned Gemini model this key can call at study scale on the free
# tier (Pro-class models return 429 immediately - no usable free quota).
# Document this substitution in Chapter 3/Table 3.3 when finalising.
MODELS = {
    "gpt-4": {"provider": "openai", "model_id": "gpt-4o"},
    "llama-3-70b": {"provider": "groq", "model_id": "llama-3.3-70b-versatile"},
    "gemini-3.1-flash-lite": {"provider": "gemini", "model_id": "gemini-3.1-flash-lite"},
}

# Experimental design constants (Chapter 3, Section 3.7)
TEMPERATURE = 0
FEW_SHOT_EXAMPLES = 4  # within the proposal's "three to five" range
INSTANCES_PER_TASK = 100  # per Chapter 3: "100-200 instances per task category"
RANDOM_SEED = 42

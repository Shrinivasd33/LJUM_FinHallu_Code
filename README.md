# Fin-Hallu Score Experiment Pipeline

This folder implements the five-phase experimental design from Chapter 3 of
the thesis ("Beyond Detection: A Diagnostic Taxonomy and Mitigation Policy
for Financial LLM Hallucinations"). Everything here has been built and,
where possible, actually tested - see "What has been verified" below for
exactly what's proven to work vs. what still needs your API keys.

## Environment

This machine already has a conda environment called **`ai_env`** with almost
everything needed pre-installed (pandas, numpy, scipy, scikit-learn,
requests, python-dotenv, datasets/huggingface_hub, openai). I additionally
installed `groq`, `google-genai`, and `pytest` into it. **Use this
environment for everything below** - do not use the plain `python` on PATH
(a separate, empty Python 3.11 install), or you'll hit missing-package errors.

```powershell
# Always invoke Python like this (or activate the env first: conda activate ai_env)
C:\Users\Sonali\anaconda3\envs\ai_env\python.exe <script>
```

If you ever need to reinstall everything from scratch elsewhere (e.g. a
fresh machine or Google Colab):
```bash
pip install -r requirements.txt
```

## One-time setup: API keys

1. Copy `config/.env.example` to `config/.env`
2. Fill in your keys:
   - `OPENAI_API_KEY` - from https://platform.openai.com/api-keys (**pay-per-use**.
     Registry uses `gpt-4o`, not classic `gpt-4` - verified that classic GPT-4
     pricing at this study's call volume would exceed $100, well past the
     proposal's under-$5 estimate; a real paid test batch on the most
     token-expensive task confirmed gpt-4o's projected total at ~$22)
   - `GROQ_API_KEY` - from https://console.groq.com/keys (free tier, rate-limited).
     Serves Llama-3.3-70B-Versatile.
   - `GOOGLE_API_KEY` - from https://aistudio.google.com/apikey (free tier, rate-limited).
     Registry uses `gemini-3.5-flash`, not `gemini-1.5-pro` - Google withdrew
     API access to gemini-1.5-pro for new projects entirely (404 "no longer
     available", verified live) before this was set up; gemini-3.5-flash is
     the current stable model reachable at this study's scale on the free
     tier (Pro-class models return no usable free quota).
   - `HF_TOKEN` - optional, only needed to unlock 3 gated FinBen datasets (see below)

`config/.env` is in `.gitignore` and is never read by anything except your
own local scripts - it does not get sent anywhere except directly to each
API provider when you make a call.

## One-time setup: gated datasets (optional but recommended)

Three of the eight FinBen task datasets require a free, one-time
click-through agreement (not a paywall - just responsible-use terms):

| Dataset | Fin-Hallu dimension | Link |
|---|---|---|
| `TheFinAI/flare-finqa` | NPI (numerical QA) | https://huggingface.co/datasets/TheFinAI/flare-finqa |
| `TheFinAI/flare-cd` | CRI (causal detection) | https://huggingface.co/datasets/TheFinAI/flare-cd |
| `TheFinAI/flare-ectsum` | TAI (earnings call summarisation) | https://huggingface.co/datasets/TheFinAI/flare-ectsum |

**Without this step, CRI has zero data at all** - both of FinBen's causal
tasks are gated, with no open substitute. To unlock:
1. Create a free account at https://huggingface.co/join
2. Visit each link above (while logged in) and click "Agree and access repository"
3. Generate a read token at https://huggingface.co/settings/tokens
4. Put it in `config/.env` as `HF_TOKEN=hf_xxxxxxxx`

## Running the pipeline

```powershell
$PY = "C:\Users\Sonali\anaconda3\envs\ai_env\python.exe"

# Phase 1: download + preprocess all data (no API keys needed, free, ~1 min)
& $PY src/data_loading.py
& $PY src/preprocessing.py

# Phase 2+3: query the models (COSTS MONEY for GPT-4 - start small!)
# Sanity-check with --limit first:
& $PY src/run_experiments.py --dimension NPI --models gpt-4 --condition zero_shot --limit 5

# Once happy, run the real thing per dimension/model/condition, e.g.:
& $PY src/run_experiments.py --dimension NPI --models gpt-4,llama-3-70b,gemini-3.5-flash --condition zero_shot
& $PY src/run_experiments.py --dimension NPI --models gpt-4,llama-3-70b,gemini-3.5-flash --condition few_shot
& $PY src/run_experiments.py --dimension NPI --models gpt-4,llama-3-70b,gemini-3.5-flash --condition rag
# repeat --dimension for TAI, EFS, CRI

# Phase 4+5: score everything collected so far and build the deployment matrix
# (free, instant, safe to re-run any time as more results come in)
& $PY src/run_scoring.py
```

`run_experiments.py` is **resumable** - if it's interrupted (rate limit,
laptop sleep, crash), just re-run the same command; it skips instances
already recorded in `results/*.jsonl`.

## What has been verified (2026-07-19)

Actually run and passing on this machine, no fabricated results:

- **Phase 1 data loading**: real download of Financial PhraseBank (2,264
  sentences) and 7 FinBen task datasets (FinQA, FNXL, FOMC, NER, FiNER-Ord,
  FinRED, CausalDetection - all unlocked with `HF_TOKEN`). Verified column
  schemas match what the code expects, including the per-task `answer_format`
  field described below.
- **Phase 1 preprocessing**: verified zero-shot/few-shot prompts build
  correctly from real downloaded data, with deterministic sampling.
- **Phase 4 scoring** (`fin_hallu_score.py`): 30/30 unit tests pass -
  numeric normalisation (handles `$1,234.50` vs `1234.5`), single-label
  accuracy, BIO-sequence tag-pair F1, entity-list pair F1, triplet F1,
  composite score formula (both equal-weight and the 35/15/35/15 weighted
  variant), Cohen's Kappa.
- **Phase 5 stats + matrix** (`stats_analysis.py`, `deployment_matrix.py`):
  7/7 unit tests pass - paired t-test/Wilcoxon selection logic, ANOVA,
  Strategic Deployment Matrix construction. Also smoke-tested end-to-end
  with synthetic data simulating exactly the "RAG helps one model, hurts
  another" pattern central to RQ2/RQ3 - the pipeline correctly detected and
  reported it.
- **Model wrappers** (`models.py`): verified they construct correctly and
  fail with clear, actionable errors when keys are missing. **Now verified
  against all three real APIs (2026-07-26)** with real, billed/rate-limited
  calls through the full pipeline (prompt build -> model call -> scoring):
  Llama-3.3-70B-Versatile (Groq, free), Gemini 3.5 Flash (Google, free), and
  GPT-4o (OpenAI, paid - a real 45-call test batch on the most
  token-expensive task confirmed the projected ~$22 total cost before the
  full run was launched). Full experiment runs for all three models are
  resumable and in progress via `run_llama3_full.ps1`, `run_gemini_full.ps1`,
  and `run_gpt4_full.ps1` in this folder.
  **Known footgun**: `get_model(name)` does NOT read `config/task_config.py`'s
  `MODELS` dict - it calls `registry[name]()` with no arguments, so only each
  wrapper class's own `__init__` default `model_id` in `models.py` controls
  which model is actually called. Change the wrapper default directly if you
  need to switch models; editing `task_config.py`'s `MODELS` dict alone has
  no runtime effect (it documents intent only).

Run the test suite yourself any time:
```powershell
& $PY -m pytest tests/ -v
```

## Design note: per-task `answer_format`, and using FinBen's native prompts

Inspecting the real FinBen ground-truth schemas directly (not just the
proposal's description of them) turned up two things that reshaped this
pipeline:

1. **FinBen tasks are far more heterogeneous than "short answer".** FNXL,
   FiNER-Ord and CausalDetection are full per-token BIO sequence-tagging
   tasks; NER is an entity-list task; FinRED is a triplet-extraction task.
   Scoring all of these with one instruction/scorer per *dimension* (as
   originally planned) would have silently produced meaningless results -
   e.g. asking a model to "reply with the number only" on a task whose
   ground truth is a 40-token tag sequence. Every task in
   `config/task_config.py` therefore carries an explicit `answer_format`
   (`numeric` / `single_label` / `bio_sequence` / `entity_list` /
   `triplet_list`), and `fin_hallu_score.py` dispatches scoring on that
   field per-row rather than per-dimension. This field is threaded all the
   way through: `data_loading.py` sets it, `preprocessing.py` and
   `run_experiments.py` carry it into `results/*.jsonl`, and
   `run_scoring.py` reads it back out for scoring - so it's always the
   actual task format doing the scoring, never an assumption from the
   dimension name.
2. **FinBen's native `query` field is already a complete, self-sufficient
   prompt** - checked directly for every task type. It already contains the
   precise instruction, the exact label/tag vocabulary (e.g. FNXL's ~100
   XBRL concept names), the format spec, and the context/question text,
   ending in a ready-to-complete "...Answer:". Writing custom per-task
   instructions on top of this would only risk duplicating or contradicting
   something FinBen already states more precisely. So `preprocessing.py`
   builds prompts directly from `query` plus one generic one-line persona
   preamble - it does not layer on dimension- or task-specific instructions.
   The one exception is **PhraseBank**, which is raw `sentence@label` data
   with no native instruction, so it gets one small task-specific
   instruction of its own (`PHRASEBANK_INSTRUCTION` in `task_config.py`).

## Known methodology deviations from the proposal (disclosed, not hidden)

1. **NPI excludes TAT-QA.** Its ground truth mixes numeric answers,
   date/year lists, and free-text explanatory spans within the same task,
   with no reliable field distinguishing them in advance. Rather than
   silently score free-text answers as numeric failures, it's left out;
   FinQA (fully unlocked) and FNXL cover NPI instead.
2. **TAI uses FOMC (hawkish/dovish/neutral stance classification) as its
   only source**, not a dedicated "temporal extraction" task - no such
   FinBen dataset exists. ECTSum was considered but excluded: its ground
   truth is a binary sentence-inclusion sequence for extractive
   summarisation, not a temporal/date label, so it doesn't actually measure
   "temporal alignment" as Chapter 3 defines it. Discuss with your
   supervisor whether FOMC alone is an adequate operationalisation, or
   whether this should be flagged as a scope limitation in Chapter 5.
3. **CRI's CausalDetection task is a cause/effect span-extraction task, not
   a binary causal/non-causal classifier** - direct inspection of all 226
   examples showed every single one contains at least one CAUSE/EFFECT
   span (there is no "non-causal" class in this dataset). CRI is scored as
   BIO-sequence tag-pair F1 accordingly, matching what the data actually is.
4. **Two of the three proposal-stage model names were substituted before
   execution, both provider-forced rather than chosen for convenience.**
   Classic GPT-4 pricing at this study's call volume was verified to exceed
   $100 against the proposal's under-$5 estimate, so `gpt-4o` was used
   instead - OpenAI's current model in the GPT-4 family. Google withdrew API
   access to `gemini-1.5-pro` for new projects entirely (a live-verified 404
   "no longer available") before experiments began, so `gemini-3.5-flash`
   was used instead - the current stable Gemini model reachable at this
   study's scale on the free tier. Both stay within the model family named
   in the approved proposal, and both are a real-time instance of the
   model-version-drift risk the thesis's Scope section (Chapter 1, Section
   1.5) already anticipates. See `config/task_config.py`'s `MODELS` comment
   and `src/models.py`'s wrapper defaults for the exact versions in use.
5. **RAG's retrieval corpus** is built from other instances' context
   passages within the same task (plus a small hand-written financial
   glossary), rather than a separately licensed corpus of annual reports -
   this keeps everything runnable on free-tier resources with no extra
   licensing questions, and is a standard simplification for this kind of
   ablation study, but it is a real methodological choice worth a sentence
   in Chapter 3.

## Folder structure

```
ExperienmentsandCoding/
├── config/
│   ├── task_config.py      # dataset repo names, model registry, constants
│   ├── .env.example        # copy to .env and fill in your keys
│   └── .env                # (you create this - gitignored)
├── src/
│   ├── data_loading.py     # Phase 1a: download FinBen + PhraseBank
│   ├── preprocessing.py    # Phase 1b: build prompts, sample instances
│   ├── models.py           # Phase 2/3: GPT-4 / Llama-3 / Gemini wrappers
│   ├── rag.py              # Phase 3: TF-IDF retrieval + prompt augmentation
│   ├── run_experiments.py  # Phase 2/3 orchestration (the part that costs money)
│   ├── fin_hallu_score.py  # Phase 4: NPI/TAI/EFS/CRI scoring
│   ├── stats_analysis.py   # Phase 5: significance tests
│   ├── deployment_matrix.py # Phase 5: Strategic Deployment Matrix
│   └── run_scoring.py      # Phase 4/5 orchestration (free, run any time)
├── tests/                  # pytest suite - 37 tests, all passing
├── data/{raw,processed}/   # downloaded + preprocessed data (gitignored)
├── results/                # experiment outputs (gitignored)
└── requirements.txt
```

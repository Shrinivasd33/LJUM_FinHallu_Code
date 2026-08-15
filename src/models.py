"""
Phase 2/3: Unified model API wrapper for GPT-4, Llama-3 (via Groq), and
Gemini-1.5, per Chapter 3, Table 3.3 and Section 3.7 of the thesis.

Requires API keys in config/.env (see config/.env.example):
  OPENAI_API_KEY=...
  GROQ_API_KEY=...
  GOOGLE_API_KEY=...

All three wrappers share one interface: `.generate(prompt: str) -> str`.
Temperature is fixed to 0 throughout (Chapter 3, Section 3.7 reproducibility
rule). Retries with exponential backoff handle transient rate-limit errors,
which are common on the Gemini and Groq free tiers.

This module makes NO API calls on import - nothing runs or costs money
until you explicitly call .generate() from an orchestration script.
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "config" / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEMPERATURE = 0
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2
# Per-attempt network timeout, seconds. Without an explicit value the openai
# SDK defaults to 600s, which let a single hung connection block a worker
# for up to ~50 minutes across all 5 retry attempts before generate() gave
# up (observed twice on DeepInfra, 2026-08-15). 60s is generous for a single
# completion call at this study's prompt lengths while failing fast enough
# for the existing exponential-backoff retry loop to actually do its job.
REQUEST_TIMEOUT_SECONDS = 60


class ModelWrapper(ABC):
    name: str

    @abstractmethod
    def _call(self, prompt: str) -> str:
        ...

    def generate(self, prompt: str) -> str | None:
        """Call the model with retry/backoff. Returns None (not an exception)
        on final failure, so callers can record it as a parse-error / null
        output per Chapter 3.4's error-handling rule, rather than crashing
        the whole batch run."""
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call(prompt)
            except Exception as e:
                last_err = e
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "%s: attempt %d/%d failed (%s: %s) - retrying in %ds",
                    self.name, attempt, MAX_RETRIES, type(e).__name__, e, wait,
                )
                time.sleep(wait)
        logger.error("%s: all %d attempts failed, giving up. Last error: %s", self.name, MAX_RETRIES, last_err)
        return None


class GPT4Wrapper(ModelWrapper):
    name = "gpt-4"

    def __init__(self, model_id: str = "gpt-4o"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in config/.env")
        self.client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self.model_id = model_id
        self.provider = "openai"

    def _call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()


class Llama3Wrapper(ModelWrapper):
    # 2026-08: THIRD provider-level substitution this study, same class of
    # reason as the two Gemini swaps (Chapter 3, Section 3.6) - Groq's free
    # tier caps llama-3.3-70b-versatile at 100,000 tokens/day (verified live
    # via the 429 error body), and Groq's paid Developer tier is currently
    # closed to new signups ("temporarily unavailable due to high demand",
    # confirmed on console.groq.com/settings/billing) - not a choice, a
    # provider-side fact. This is EXPLICITLY within the approved proposal's
    # own risk register (Risk 3: "LLM API model deprecation mid-experiment",
    # contingency: swap provider/model if one becomes unsupported) and
    # Table 6, which already named TWO acceptable access methods for this
    # model ("HuggingFace Inference API or Groq free tier") - this is a
    # smaller, more conservative move than that pre-approved contingency
    # even allows (same exact model checkpoint, Llama-3.3-70B, just hosted by
    # a different provider - DeepInfra, OpenAI-compatible API - rather than a
    # full model-family swap to Mistral-7B as the risk register permits).
    # Preference order: Cerebras > DeepInfra > Groq. FINAL DECISION 2026-08:
    # DeepInfra is active (paid balance funded by the user) after Groq's own
    # deprecations page confirmed llama-3.3-70b-versatile stops being served
    # "by August 2026" (checked live, still responding as of this decision,
    # but the shutdown could land at any time - not a hypothetical risk).
    # Groq's own recommended replacements for that deprecation are
    # gpt-oss-120b or qwen3.6-27b - NEITHER is a Llama model - so continuing
    # to wait on Groq risked being forced into an actual model-family swap
    # with no notice. Moving to DeepInfra proactively keeps the exact same
    # model (meta-llama/Llama-3.3-70B-Instruct, verified identical to what
    # Groq serves - Meta released Llama 3.3 70B as a single Instruct-only
    # checkpoint, no alternate base variant either provider could be using
    # instead) - the smallest possible deviation, already pre-approved by
    # the proposal's Table 6 ("HuggingFace Inference API or Groq free
    # tier") and risk register (Risk 3: model/provider substitution
    # contingency). Cerebras was also investigated and rejected (this
    # account's live model list has no Llama model at all - would have been
    # a real model-family deviation, unlike DeepInfra). Groq remains coded
    # as the final fallback (not deleted) in case DeepInfra becomes
    # unavailable later.
    name = "llama-3-70b"

    def __init__(self, model_id: str = "llama-3.3-70b-versatile"):
        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if cerebras_key:
            from openai import OpenAI  # Cerebras exposes an OpenAI-compatible API
            self.client = OpenAI(api_key=cerebras_key, base_url="https://api.cerebras.ai/v1", timeout=REQUEST_TIMEOUT_SECONDS)
            self.model_id = "llama-3.3-70b"
            self.provider = "cerebras"
        elif deepinfra_key:
            from openai import OpenAI  # DeepInfra exposes an OpenAI-compatible API
            # 2026-08-15: explicit timeout added after two separate hangs
            # (EFS zero-shot, both times) where a call neither succeeded nor
            # raised an exception for 30-40+ minutes - the openai SDK's
            # default timeout is 600s per attempt with no override here
            # previously, so 5 retry attempts could hang for the better part
            # of an hour before finally giving up. A short explicit timeout
            # lets the existing retry/backoff logic in generate() do its job
            # promptly instead of stalling silently.
            self.client = OpenAI(api_key=deepinfra_key, base_url="https://api.deepinfra.com/v1/openai", timeout=REQUEST_TIMEOUT_SECONDS)
            # 2026-08-15: corrected to the model ID DeepInfra actually serves and
            # bills under (confirmed live via GET /v1/openai/models and the
            # account's own usage dashboard) - the plain "-Instruct" ID this
            # wrapper previously requested does not exist as a separate
            # deployment on DeepInfra at all; it was being silently aliased to
            # this exact "-Turbo" ID the whole time, so this is a documentation
            # correction to match observed reality, not a behaviour change.
            # "-Turbo" is DeepInfra's FP8-quantized serving of the checkpoint,
            # not a byte-identical FP16 reference - disclosed as a limitation in
            # Chapter 5, Section 5.8 and Chapter 3, Section 3.6.
            self.model_id = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            self.provider = "deepinfra"
        elif groq_key:
            from groq import Groq
            self.client = Groq(api_key=groq_key, timeout=REQUEST_TIMEOUT_SECONDS)
            self.model_id = model_id
            self.provider = "groq"
        else:
            raise RuntimeError("None of CEREBRAS_API_KEY, DEEPINFRA_API_KEY, GROQ_API_KEY set in config/.env")

    def _call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()


class GeminiWrapper(ModelWrapper):
    # 2026-07-26: TWO substitutions were needed. (1) gemini-1.5-pro was
    # retired by Google for new API projects (404 on generateContent).
    # (2) The first replacement, gemini-3.5-flash, turned out to carry only a
    # 20-requests/DAY free-tier quota on this project - verified via the 429
    # error body's quotaValue field, which silently nulled ~70% of the first
    # batch of outputs once the retry budget was exhausted (a null result is
    # NOT the same as a wrong answer; it was cleaned out of results/ before
    # continuing). gemini-3.1-flash-lite was chosen after directly verifying
    # it survives a 10-call burst with no 429, giving usable throughput at
    # this study's scale on the free tier.
    name = "gemini-3.1-flash-lite"

    def __init__(self, model_id: str = "gemini-3.1-flash-lite"):
        # Uses the current `google-genai` SDK - the older `google.generativeai`
        # package was fully deprecated by Google and should not be used for
        # new work.
        from google import genai
        from google.genai import types
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in config/.env")
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000))
        self.model_id = model_id
        self.provider = "google"
        self._types = types

    def _call(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=self._types.GenerateContentConfig(temperature=TEMPERATURE),
        )
        return resp.text.strip()


def get_model(name: str) -> ModelWrapper:
    """Factory: get_model('gpt-4' | 'llama-3-70b' | 'gemini-3.1-flash-lite')"""
    registry = {
        "gpt-4": GPT4Wrapper,
        "llama-3-70b": Llama3Wrapper,
        "gemini-3.1-flash-lite": GeminiWrapper,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}")
    return registry[name]()


if __name__ == "__main__":
    # Smoke test: verify each wrapper can be constructed (checks keys are
    # present) WITHOUT making a real API call, so this is free to run.
    for name in ["gpt-4", "llama-3-70b", "gemini-3.1-flash-lite"]:
        try:
            get_model(name)
            print(f"{name}: OK - API key found, wrapper constructed successfully.")
        except RuntimeError as e:
            print(f"{name}: NOT READY - {e}")
        except Exception as e:
            print(f"{name}: IMPORT/SETUP ERROR - {type(e).__name__}: {e}")

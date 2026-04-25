"""Model registry for the EnergyX LLM benchmark.

Two provider families: Cerebras (cloud, wafer-scale inference) and Ollama (local).
All models are accessed via the existing LangChain factory pattern — this registry
drives which model/provider pair is injected per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    model_id: str        # canonical slug used in result filenames
    display_name: str    # human-readable label for tables
    provider: str        # "cerebras" | "ollama"
    model_name: str      # name passed to the API / Ollama pull
    tier: str            # "small" | "medium" | "large"
    context_window: int  # tokens
    uses_thinking: bool = False  # model emits <think>...</think> before answer
    thinking_token_budget: int = 0  # extra tokens to allocate for the think block
    notes: str = ""


# ---------------------------------------------------------------------------
# Cerebras (cloud, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
CEREBRAS_MODELS: List[ModelConfig] = [
    ModelConfig(
        model_id="cerebras/llama3.1-8b",
        display_name="Llama 3.1 8B (Cerebras)",
        provider="cerebras",
        model_name="llama3.1-8b",
        tier="small",
        context_window=8192,
        notes="Production default; ~500 tok/s on Cerebras wafer silicon",
    ),
    ModelConfig(
        model_id="cerebras/gpt-oss-120b",
        display_name="GPT-OSS 120B (Cerebras)",
        provider="cerebras",
        model_name="gpt-oss-120b",
        tier="large",
        context_window=8192,
        notes="Open-source 120B model on Cerebras wafer; primary large-model baseline",
    ),
]

# ---------------------------------------------------------------------------
# Ollama (local inference)
# ---------------------------------------------------------------------------
OLLAMA_MODELS: List[ModelConfig] = [
    ModelConfig(
        model_id="ollama/deepseek-r1:1.5b",
        display_name="DeepSeek R1 1.5B (Ollama)",
        provider="ollama",
        model_name="deepseek-r1:1.5b",
        tier="small",
        context_window=4096,
        uses_thinking=True,
        thinking_token_budget=400,
        notes="Reasoning-distilled; emits <think> chain before answer — strip before eval",
    ),
    ModelConfig(
        model_id="ollama/llama3.2:3b",
        display_name="Llama 3.2 3B (Ollama)",
        provider="ollama",
        model_name="llama3.2:3b",
        tier="small",
        context_window=4096,
        notes="Meta compact model; tests whether 3B is viable for agentic routing",
    ),
    ModelConfig(
        model_id="ollama/llama3.1:8b",
        display_name="Llama 3.1 8B (Ollama)",
        provider="ollama",
        model_name="llama3.1:latest",
        tier="small",
        context_window=8192,
        notes="Same family as Cerebras baseline; isolates cloud vs local inference",
    ),
    ModelConfig(
        model_id="ollama/qwen2.5:7b-instruct",
        display_name="Qwen 2.5 7B Instruct (Ollama)",
        provider="ollama",
        model_name="qwen2.5:7b-instruct",
        tier="small",
        context_window=32768,
        notes="Strong structured-output and tool-use for size; long context",
    ),
    ModelConfig(
        model_id="ollama/mistral:7b-instruct",
        display_name="Mistral 7B Instruct (Ollama)",
        provider="ollama",
        model_name="mistral:7b-instruct",
        tier="small",
        context_window=8192,
        notes="European instruction-tuned model; strong at precise JSON and tool use",
    ),
]

# Models to be added once pulled locally (run: ollama pull <model_name>)
OLLAMA_MODELS_PENDING: List[ModelConfig] = [
    ModelConfig(model_id="ollama/mistral:7b",    display_name="Mistral 7B (Ollama)",    provider="ollama", model_name="mistral:7b",    tier="small",  context_window=8192,  notes="Pending pull"),
    ModelConfig(model_id="ollama/qwen2.5:7b",   display_name="Qwen 2.5 7B (Ollama)",   provider="ollama", model_name="qwen2.5:7b",   tier="small",  context_window=32768, notes="Pending pull"),
    ModelConfig(model_id="ollama/gemma3:9b",    display_name="Gemma 3 9B (Ollama)",    provider="ollama", model_name="gemma3:9b",    tier="small",  context_window=8192,  notes="Pending pull"),
    ModelConfig(model_id="ollama/phi4:14b",     display_name="Phi 4 14B (Ollama)",     provider="ollama", model_name="phi4:14b",     tier="medium", context_window=16384, notes="Pending pull"),
]

ALL_MODELS: List[ModelConfig] = CEREBRAS_MODELS + OLLAMA_MODELS


def get_model(model_id: str) -> Optional[ModelConfig]:
    for m in ALL_MODELS:
        if m.model_id == model_id:
            return m
    return None


def effective_max_tokens(model_cfg: ModelConfig, requested: int) -> int:
    """Add thinking budget on top of the requested token cap for reasoning models."""
    return requested + model_cfg.thinking_token_budget if model_cfg.uses_thinking else requested


def get_llm_for_model(model_cfg: ModelConfig, temperature: float = 0.1, max_tokens: int = 200):
    """Build a LangChain LLM instance for the given model config.

    max_tokens is the budget for the *answer* portion. For thinking models,
    effective_max_tokens() automatically adds the thinking_token_budget on top
    so the model has room to reason before producing its answer.

    Per-phase answer budgets:
      routing    →  10  (single intent word)
      tool_call  → 200  (tool name + args JSON)
      synthesis  → 150  (2-sentence answer)
      judge      →  20  (YES/NO + brief reason)
    """
    from tinyts.config import settings
    actual_max = effective_max_tokens(model_cfg, max_tokens)

    if model_cfg.provider == "cerebras":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=settings.cerebras_base_url,
            api_key=settings.cerebras_api_key,
            model=model_cfg.model_name,
            temperature=temperature,
            max_tokens=actual_max,
        )
    else:  # ollama
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model_cfg.model_name,
            temperature=temperature,
            num_predict=actual_max,
        )


import re as _re
_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output.

    Returns the stripped answer text, stripped of leading/trailing whitespace.
    If no think block is present, returns the original text unchanged.
    """
    return _THINK_RE.sub("", text).strip()

"""
src/agent.py
=============
LLM agent for context-aware Q&A on MMM results.

The agent can ONLY answer from two sources:
  1. docs/methodology.md — explains what MMM is, how it works, limitations
  2. Model results — the actual numbers from the PyMC posterior

If a question falls outside these sources, the agent must say so.
Uses OpenRouter API (OpenAI-compatible) with free-tier models.
"""

import os
import json
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI


# ── Load environment and config ──────────────────────────────
load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_METHODOLOGY_PATH = Path(__file__).parent.parent / "docs" / "methodology.md"


def _load_config() -> dict:
    """Load agent config from config.yaml."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f).get("agent", {})
    return {}


def _load_methodology() -> str:
    """Load methodology.md — the agent's knowledge base."""
    if _METHODOLOGY_PATH.exists():
        return _METHODOLOGY_PATH.read_text(encoding="utf-8")
    return "No methodology document found."


def build_system_prompt() -> str:
    """
    Build the system prompt that grounds the agent.

    Includes:
      - Role definition
      - Full methodology.md content
      - Strict instruction to answer only from context
      - Refusal template for unknown questions
    """
    methodology = _load_methodology()

    return f"""You are an expert Marketing Mix Modelling analyst. You help users
understand their MMM results and make budget decisions.

You have TWO sources of knowledge and you must ONLY answer from these:

1. METHODOLOGY REFERENCE (how the model works):
{methodology}

2. MODEL RESULTS (the actual numbers — provided in each user message)

STRICT RULES:
- Answer ONLY from the methodology reference and the model results provided.
- If a question is outside your context, say: "I can only answer based on the
  analyses run on your data. This question falls outside the available context."
- Never invent numbers. Only cite numbers from the model results.
- When discussing results, always mention credible intervals — not just point
  estimates. Say "Search ROAS is 4.2x (94% CI: 3.1–5.3)" not just "4.2x".
- If the user asks about a channel not in the results, say so.
- You cannot run new analyses, retrain the model, or access external data.
- Be direct and concise. Write for a marketing manager, not a statistician."""


def build_user_context(
    model_results: dict[str, Any],
    optimiser_results: Optional[dict[str, Any]] = None,
) -> str:
    """
    Convert model results into a readable context block for the LLM.

    Parameters
    ----------
    model_results : dict
        Output from extract_model_results().
    optimiser_results : dict, optional
        Output from optimise_budget().

    Returns
    -------
    str — formatted context to prepend to the user's question.
    """
    lines = ["=== MODEL RESULTS ===\n"]

    # ── Channel contributions ────────────────────────────────────
    lines.append("CHANNEL CONTRIBUTIONS (% of total revenue):")
    base = model_results.get("base_revenue", {})
    lines.append(
        f"  Base/organic: {base.get('share_pct', 'N/A')}% "
        f"(${base.get('mean', 'N/A'):,} total, "
        f"94% CI: ${base.get('ci_lower', 'N/A'):,}–${base.get('ci_upper', 'N/A'):,})"
    )
    for ch in model_results.get("channel_contributions", []):
        lines.append(
            f"  {ch['channel']}: {ch['share_pct']}% "
            f"(${ch['mean']:,} total, "
            f"94% CI: ${ch['ci_lower']:,}–${ch['ci_upper']:,})"
        )

    # ── ROAS ─────────────────────────────────────────────────────
    lines.append("\nROAS (Return on Ad Spend) per channel:")
    for r in model_results.get("roas", []):
        lines.append(
            f"  {r['channel']}: {r['mean']}x "
            f"(94% CI: {r['ci_lower']}–{r['ci_upper']})"
        )

    # ── Adstock ──────────────────────────────────────────────────
    lines.append("\nADSTOCK DECAY RATES (learned from data):")
    for a in model_results.get("adstock_decay", []):
        lines.append(
            f"  {a['channel']}: {a['mean']} "
            f"(94% CI: {a['ci_lower']}–{a['ci_upper']})"
        )

    # ── Saturation ───────────────────────────────────────────────
    lines.append("\nSATURATION PARAMETERS (learned from data):")
    for s in model_results.get("saturation_params", []):
        lines.append(
            f"  {s['channel']}: lambda={s['lam_mean']} "
            f"(94% CI: {s['lam_ci_lower']}–{s['lam_ci_upper']}), "
            f"beta={s['beta_mean']} "
            f"(94% CI: {s['beta_ci_lower']}–{s['beta_ci_upper']})"
        )

    # ── Optimiser ────────────────────────────────────────────────
    if optimiser_results:
        lines.append("\nBUDGET OPTIMISATION:")
        lines.append(
            f"  Total weekly budget: ${optimiser_results['total_budget']:,}"
        )
        lines.append(
            f"  Predicted revenue uplift from reallocation: "
            f"{optimiser_results['predicted_uplift_pct']}%"
        )
        lines.append("  Recommended changes:")
        for a in optimiser_results.get("allocations", []):
            direction = "increase" if a["change"] > 0 else "decrease"
            lines.append(
                f"    {a['channel']}: ${a['current_spend']:,} → "
                f"${a['optimal_spend']:,} ({direction} ${abs(a['change']):,})"
            )

    return "\n".join(lines)


def create_agent_messages(
    question: str,
    model_results: dict[str, Any],
    optimiser_results: Optional[dict[str, Any]] = None,
    history: Optional[list[dict]] = None,
) -> list[dict[str, str]]:
    """
    Build the full message array for the OpenRouter API call.

    Parameters
    ----------
    question : str
        The user's current question.
    model_results : dict
        Output from extract_model_results().
    optimiser_results : dict, optional
        Output from optimise_budget().
    history : list[dict], optional
        Previous conversation turns [{"role": "user"|"assistant", "content": "..."}].

    Returns
    -------
    list[dict] — messages array ready for the OpenAI-compatible API.
    """
    messages = [{"role": "system", "content": build_system_prompt()}]

    # ── Add conversation history if present ──────────────────────
    if history:
        messages.extend(history)

    # ── Build user message with context + question ───────────────
    context = build_user_context(model_results, optimiser_results)
    user_content = f"{context}\n\n=== USER QUESTION ===\n{question}"
    messages.append({"role": "user", "content": user_content})

    return messages


def ask_agent(
    question: str,
    model_results: dict[str, Any],
    optimiser_results: Optional[dict[str, Any]] = None,
    history: Optional[list[dict]] = None,
) -> str:
    """
    Send a question to the LLM via OpenRouter and return the response.

    Parameters
    ----------
    question : str
        The user's question about the MMM results.
    model_results : dict
        Output from extract_model_results().
    optimiser_results : dict, optional
        Output from optimise_budget().
    history : list[dict], optional
        Previous conversation turns.

    Returns
    -------
    str — the agent's response text.
    """
    config = _load_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key or api_key == "your_api_key_here":
        return (
            "OpenRouter API key not configured. "
            "Please add your key to the .env file."
        )

    # ── Build messages ───────────────────────────────────────────
    messages = create_agent_messages(
        question=question,
        model_results=model_results,
        optimiser_results=optimiser_results,
        history=history,
    )

    # ── Call OpenRouter (OpenAI-compatible API) ──────────────────
    client = OpenAI(
        base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model=config.get("model", "meta-llama/llama-3.1-70b-instruct:free"),
            messages=messages,
            max_tokens=config.get("max_tokens", 2000),
            temperature=config.get("temperature", 0.2),
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

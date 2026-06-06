"""
tests/test_agent.py
====================
Goal-framed tests for the LLM agent module.

These tests verify:
  - Agent builds correct prompt from model results + methodology
  - Agent refuses questions outside its context
  - System prompt enforces grounded answers
  - Works when model results are provided
  - Works for follow-up questions

NOTE: Tests mock the OpenRouter API call — no real API key needed.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agent import build_system_prompt, build_user_context, create_agent_messages


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_results():
    """Simulated extracted model results."""
    return {
        "channel_contributions": [
            {"channel": "spend_search", "mean": 85000, "ci_lower": 70000,
             "ci_upper": 100000, "share_pct": 18.3},
            {"channel": "spend_tv", "mean": 45000, "ci_lower": 30000,
             "ci_upper": 60000, "share_pct": 9.7},
        ],
        "roas": [
            {"channel": "spend_search", "mean": 4.2, "ci_lower": 3.1, "ci_upper": 5.3},
            {"channel": "spend_tv", "mean": 1.6, "ci_lower": 0.8, "ci_upper": 2.4},
        ],
        "base_revenue": {"mean": 320000, "ci_lower": 290000,
                         "ci_upper": 350000, "share_pct": 69.0},
        "adstock_decay": [
            {"channel": "spend_search", "mean": 0.28, "ci_lower": 0.15, "ci_upper": 0.41},
            {"channel": "spend_tv", "mean": 0.65, "ci_lower": 0.50, "ci_upper": 0.80},
        ],
        "saturation_params": [
            {"channel": "spend_search", "lam_mean": 2.1, "lam_ci_lower": 1.2,
             "lam_ci_upper": 3.0, "beta_mean": 0.5, "beta_ci_lower": 0.3,
             "beta_ci_upper": 0.7},
            {"channel": "spend_tv", "lam_mean": 1.5, "lam_ci_lower": 0.8,
             "lam_ci_upper": 2.2, "beta_mean": 0.4, "beta_ci_lower": 0.2,
             "beta_ci_upper": 0.6},
        ],
    }


@pytest.fixture
def sample_optimiser():
    """Simulated optimiser output."""
    return {
        "allocations": [
            {"channel": "spend_search", "current_spend": 3500,
             "optimal_spend": 5200, "change": 1700},
            {"channel": "spend_tv", "current_spend": 5500,
             "optimal_spend": 3800, "change": -1700},
        ],
        "total_budget": 9000,
        "predicted_uplift_pct": 12.4,
    }


# ────────────────────────────────────────────────────────────────
# GOAL: System prompt enforces grounded answers
# ────────────────────────────────────────────────────────────────

def test_system_prompt_contains_methodology():
    """System prompt must include methodology.md content."""
    prompt = build_system_prompt()
    # Must contain key methodology concepts
    assert "Marketing Mix Modelling" in prompt or "MMM" in prompt
    assert "adstock" in prompt.lower()
    assert "saturation" in prompt.lower()
    assert "Bayesian" in prompt or "bayesian" in prompt


def test_system_prompt_contains_refusal_instruction():
    """System prompt must tell the LLM to refuse out-of-context questions."""
    prompt = build_system_prompt()
    # Must contain instruction to decline unknown questions
    assert "context" in prompt.lower()
    assert any(word in prompt.lower() for word in ["cannot", "only", "outside", "don't know"])


# ────────────────────────────────────────────────────────────────
# GOAL: User context is built correctly from model results
# ────────────────────────────────────────────────────────────────

def test_user_context_includes_contributions(sample_results, sample_optimiser):
    """Context must include channel contribution numbers."""
    context = build_user_context(sample_results, sample_optimiser)
    assert "spend_search" in context
    assert "18.3" in context  # search share_pct


def test_user_context_includes_roas(sample_results, sample_optimiser):
    """Context must include ROAS values."""
    context = build_user_context(sample_results, sample_optimiser)
    assert "4.2" in context  # search ROAS mean


def test_user_context_includes_base_revenue(sample_results, sample_optimiser):
    """Context must include base revenue."""
    context = build_user_context(sample_results, sample_optimiser)
    assert "320000" in context or "320,000" in context


def test_user_context_includes_optimiser_output(sample_results, sample_optimiser):
    """Context must include budget reallocation recommendation."""
    context = build_user_context(sample_results, sample_optimiser)
    assert "12.4" in context  # uplift pct


# ────────────────────────────────────────────────────────────────
# GOAL: Message structure is correct for OpenRouter API
# ────────────────────────────────────────────────────────────────

def test_messages_have_system_and_user_roles(sample_results, sample_optimiser):
    """Messages must have system prompt + user question."""
    messages = create_agent_messages(
        question="What is the ROAS for search?",
        model_results=sample_results,
        optimiser_results=sample_optimiser,
    )
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_messages_include_context_before_question(sample_results, sample_optimiser):
    """The user message must include data context + the actual question."""
    messages = create_agent_messages(
        question="What is the ROAS for search?",
        model_results=sample_results,
        optimiser_results=sample_optimiser,
    )
    user_msg = next(m for m in messages if m["role"] == "user")
    # Must contain both the context data and the question
    assert "ROAS" in user_msg["content"]
    assert "search" in user_msg["content"].lower()


def test_followup_includes_history(sample_results, sample_optimiser):
    """Follow-up questions must carry conversation history."""
    history = [
        {"role": "user", "content": "What is search ROAS?"},
        {"role": "assistant", "content": "Search ROAS is 4.2x."},
    ]
    messages = create_agent_messages(
        question="And what about TV?",
        model_results=sample_results,
        optimiser_results=sample_optimiser,
        history=history,
    )
    # History should be in the messages
    assert len(messages) >= 4  # system + history(2) + new user

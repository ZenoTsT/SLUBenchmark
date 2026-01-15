# cost_utils.py
from typing import Any, Dict, Optional


# Prezzi per milione di token (stima locale)
PRICE_PER_M_INPUT = 0.05
PRICE_PER_M_CACHED_INPUT = 0.005
PRICE_PER_M_OUTPUT = 0.4


def extract_usage_from_response(response: Any) -> Dict[str, int]:
    """
    Estrae usage dalla Responses API in modo robusto.
    Ritorna sempre un dict con le chiavi:
      - input_tokens
      - cached_tokens
      - output_tokens
      - reasoning_tokens
      - total_tokens
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0) if input_details else 0

    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0) if output_details else 0

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost(
    usage: Dict[str, int],
    price_per_m_input: float = PRICE_PER_M_INPUT,
    price_per_m_cached_input: float = PRICE_PER_M_CACHED_INPUT,
    price_per_m_output: float = PRICE_PER_M_OUTPUT,
) -> Dict[str, float]:
    """
    Stima costo in USD a partire da usage.
    - input non-cached: (input_tokens - cached_tokens)
    - cached input: cached_tokens
    - output: output_tokens
    """
    in_tok = int(usage.get("input_tokens", 0) or 0)
    cached_tok = int(usage.get("cached_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)

    real_input_tok = max(in_tok - cached_tok, 0)

    cost_input = (real_input_tok / 1_000_000) * price_per_m_input
    cost_cached = (cached_tok / 1_000_000) * price_per_m_cached_input
    cost_output = (out_tok / 1_000_000) * price_per_m_output

    return {
        "input_cost": cost_input,
        "cached_input_cost": cost_cached,
        "output_cost": cost_output,
        "total_cost": cost_input + cost_cached + cost_output,
    }


def merge_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """
    Somma due dict usage (utile per totale).
    """
    keys = ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens"]
    return {k: int(a.get(k, 0) or 0) + int(b.get(k, 0) or 0) for k in keys}


def empty_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
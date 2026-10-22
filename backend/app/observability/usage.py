from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _int(value: Any) -> int | None:
    """Convert a token count to int without inventing missing values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float(value: Any) -> float | None:
    """Convert a provider-reported cost to float, preserving missing values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(sources: list[Mapping[str, Any]], keys: tuple[str, ...]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def extract_usage(message: Any) -> dict[str, Any]:
    """
    Extract token usage and cost reported by the LLM provider / LangChain message.

    Does not estimate or fabricate costs. OpenRouter's `usage.include` response
    commonly exposes `prompt_tokens`, `completion_tokens`, `total_tokens`, and
    `cost` in response_metadata['token_usage'] or response_metadata['usage'].
    """
    response_meta = _mapping(getattr(message, "response_metadata", None))
    usage_meta = _mapping(getattr(message, "usage_metadata", None))

    candidates: list[Mapping[str, Any]] = []
    # LangChain AIMessage.usage_metadata often uses input_tokens/output_tokens.
    if usage_meta:
        candidates.append(usage_meta)

    # OpenAI-compatible LangChain metadata may contain any of these.
    for key in ("token_usage", "usage", "tokenUsage"):
        nested = _mapping(response_meta.get(key))
        if nested:
            candidates.append(nested)

    # Some adapters place usage directly on response_metadata.
    if response_meta:
        candidates.append(response_meta)

    input_tokens = _int(_first_present(
        candidates,
        ("input_tokens", "prompt_tokens", "promptTokenCount"),
    ))
    output_tokens = _int(_first_present(
        candidates,
        ("output_tokens", "completion_tokens", "candidatesTokenCount"),
    ))
    total_tokens = _int(_first_present(
        candidates,
        ("total_tokens", "totalTokens"),
    ))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    # Provider cost may be a direct usage field or nested in response metadata.
    total_cost = _float(_first_present(
        candidates,
        ("total_cost_usd", "total_cost", "cost", "totalCost"),
    ))
    input_cost = _float(_first_present(
        candidates,
        ("input_cost_usd", "prompt_cost", "input_cost", "promptCost"),
    ))
    output_cost = _float(_first_present(
        candidates,
        ("output_cost_usd", "completion_cost", "output_cost", "completionCost"),
    ))

    # Only derive total from input/output costs if both were explicitly supplied.
    if total_cost is None and input_cost is not None and output_cost is not None:
        total_cost = input_cost + output_cost

    # Do not derive a split from total cost: provider pricing can vary by token type.
    token_source = "provider_usage" if input_tokens is not None or output_tokens is not None else "unavailable"
    if total_cost is not None:
        cost_source = "provider_cost"
    elif input_cost is not None and output_cost is not None:
        cost_source = "provider_input_output_cost"
    else:
        cost_source = "unavailable"

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "source": token_source,
        "cost_source": cost_source,
    }

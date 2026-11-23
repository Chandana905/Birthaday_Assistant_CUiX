from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


ALLOWED_INTENTS = {
    "birthday_search",
    "count_birthdays",
    "upcoming_birthdays",
    "person_place",
    "person_info",
    "unknown",
}
ALLOWED_ENTITIES = {
    "person_name",
    "relationship",
    "month",
    "relative_date",
    "relative_month",
    "place",
}
SYSTEM_INSTRUCTION = (
    "You are an NLU parser for a birthday assistant. Return JSON only with "
    "keys intent and entities. Allowed intents: birthday_search, "
    "count_birthdays, upcoming_birthdays, person_place, person_info, unknown. "
    "Allowed entities: person_name, relationship, month, relative_date, "
    "relative_month, place. Never answer the user's factual question; only "
    "parse it. Never invent people or facts. Use null for entities that are "
    "not present. month must be an integer from 1 to 12. relative_date may "
    "only be today or tomorrow. relative_month may only be this month or "
    "next month. Only extract names, relationships, and places mentioned by "
    "the user."
)
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMNLUError(ValueError):
    """Raised when an LLM request or structured response cannot be used."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "invalid_response",
        diagnostic: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.diagnostic = diagnostic or message


def get_configured_api_key() -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    api_key = api_key.strip()
    if not api_key or api_key.casefold() == "your_groq_api_key_here":
        return None
    return api_key


def diagnose_groq_error(error: Exception) -> tuple[str, str]:
    """Return an allow-listed Groq failure category and credential-safe advice."""
    error_name = type(error).__name__
    status_code = getattr(error, "status_code", None)
    if error_name in {"APITimeoutError", "TimeoutError"}:
        return "timeout", "The Groq request timed out; retry after checking connectivity."
    if error_name == "APIConnectionError" or isinstance(error, ConnectionError):
        return "network", "Could not connect to Groq; check internet, DNS, or firewall access."
    if error_name == "AuthenticationError" or status_code == 401:
        return "authentication", "Groq rejected the configured key; verify the key and account access."
    if error_name == "PermissionDeniedError" or status_code == 403:
        return "permission", "The configured account cannot access the requested Groq model."
    if error_name == "RateLimitError" or status_code == 429:
        return "rate_limit", "Groq rate limits were reached; retry later."
    if error_name == "BadRequestError" or status_code == 400:
        return "bad_request", "Groq rejected the request; check the configured model and request format."
    if status_code is not None:
        return "http_error", f"Groq returned HTTP {status_code}; check service status and model availability."
    return "client_error", f"Groq client raised {error_name}; check SDK configuration."


def _validate_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict) or set(parsed) != {"intent", "entities"}:
        raise LLMNLUError("The LLM response must contain only intent and entities.")

    intent = parsed["intent"]
    entities = parsed["entities"]
    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise LLMNLUError("The LLM returned an invalid intent.")
    if not isinstance(entities, dict):
        raise LLMNLUError("The LLM entities value must be a JSON object.")
    if not set(entities).issubset(ALLOWED_ENTITIES):
        raise LLMNLUError("The LLM returned an unsupported entity field.")

    for key, value in entities.items():
        if value is None:
            continue
        if key == "month":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 12
            ):
                raise LLMNLUError("The LLM returned an invalid month.")
        elif not isinstance(value, str) or len(value.strip()) > 120:
            raise LLMNLUError(f"The LLM returned an invalid {key} value.")
        elif key == "relative_date" and value.casefold().strip() not in {
            "today",
            "tomorrow",
        }:
            raise LLMNLUError("The LLM returned an unsupported relative date.")
        elif key == "relative_month" and value.casefold().strip() not in {
            "this month",
            "next month",
        }:
            raise LLMNLUError("The LLM returned an unsupported relative month.")

    return {
        "intent": intent,
        "entities": {
            key: value.strip() if isinstance(value, str) else value
            for key, value in entities.items()
            if value is not None
        },
    }


def parse_with_llm(message: str, client: Any = None) -> dict[str, Any]:
    """Parse a question into validated JSON using the Groq chat API."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = get_configured_api_key()
    if not api_key and client is None:
        raise LLMNLUError(
            "GROQ_API_KEY is not configured.",
            category="configuration",
        )
    if not isinstance(message, str) or not message.strip():
        raise LLMNLUError("The message is empty.")

    try:
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, timeout=15.0)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        category, diagnostic = diagnose_groq_error(exc)
        raise LLMNLUError(
            f"Groq NLU {category}: {diagnostic}",
            category=category,
            diagnostic=diagnostic,
        ) from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise LLMNLUError("The Groq response had an unexpected format.") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMNLUError("The Groq response did not contain JSON text.")
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMNLUError("The Groq response was not valid JSON.") from exc
    return _validate_response(parsed)

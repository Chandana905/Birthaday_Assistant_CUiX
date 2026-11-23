from __future__ import annotations

import re
from typing import Any

import pandas as pd

from nlu import parse


CONTEXT_INTENTS = {"birthday_search", "count_birthdays", "upcoming_birthdays"}
FILTER_FIELDS = ("month", "relationship", "place")
FOLLOWUP_PATTERN = re.compile(
    r"\b(?:what about|only those|which of them|how many\b.*\b(?:there|people)\b|"
    r"show their|list their|sort them|this person|their relationship)\b",
    re.IGNORECASE,
)
RESULT_PRONOUN_PATTERN = re.compile(
    r"\b(?:they|them|their|those|these)\b",
    re.IGNORECASE,
)
PERSON_PRONOUN_PATTERN = re.compile(
    r"\b(?:this person|they|them|their|his|her)\b",
    re.IGNORECASE,
)


def is_contextual_followup(message: str, state: dict[str, Any]) -> bool:
    has_result_context = (
        state.get("last_result_ids") is not None
        or state.get("person_id") is not None
    )
    return has_result_context and bool(
        FOLLOWUP_PATTERN.search(message)
        or RESULT_PRONOUN_PATTERN.search(message)
    )


def _state_filters(state: dict[str, Any]) -> dict[str, Any]:
    filters = dict(state.get("current_filters") or {})
    legacy_keys = {
        "month": "previous_month",
        "relationship": "previous_relationship",
        "place": "previous_place",
    }
    for key in FILTER_FIELDS:
        value = state.get(key)
        if value is None:
            value = state.get(legacy_keys[key])
        if key not in filters and value is not None:
            filters[key] = value
    return filters


def _inherit_filters(entities: dict[str, Any], filters: dict[str, Any]) -> None:
    for key in FILTER_FIELDS:
        if key not in entities and filters.get(key) is not None:
            entities[key] = filters[key]


def enrich_with_context(
    parsed: dict[str, Any],
    state: dict[str, Any],
    message: str = "",
) -> dict[str, Any]:
    """Resolve follow-ups against the saved query and result set."""
    enriched = dict(parsed)
    entities = dict(parsed.get("entities") or {})
    enriched["entities"] = entities
    enriched["context_mode"] = "base"
    if not state:
        return enriched

    filters = _state_filters(state)
    previous_intent = state.get("previous_intent") or state.get("last_intent")
    last_result_ids = state.get("last_result_ids")
    base_result_ids = state.get("base_result_ids", last_result_ids)
    has_results = last_result_ids is not None
    has_context = has_results or bool(filters) or previous_intent is not None
    has_reference = bool(
        RESULT_PRONOUN_PATTERN.search(message)
        or FOLLOWUP_PATTERN.search(message)
    )
    is_person_reference = bool(PERSON_PRONOUN_PATTERN.search(message))

    if (
        is_person_reference
        and enriched["intent"] in {"person_place", "person_info"}
        and "person_id" not in entities
    ):
        person_id = state.get("person_id")
        if not person_id and len(last_result_ids or []) == 1:
            person_id = last_result_ids[0]
        if person_id:
            entities["person_id"] = str(person_id)
            enriched["context_mode"] = "view"
            enriched["context_used"] = True
            return enriched

    if (
        has_context
        and entities.get("month") is not None
        and previous_intent in CONTEXT_INTENTS
    ):
        _inherit_filters(entities, filters)
        if enriched["intent"] == "unknown" or re.search(
            r"\bwhat about\b", message, re.IGNORECASE
        ):
            enriched["intent"] = "birthday_search"
        enriched["context_mode"] = "base"
        enriched["context_used"] = True
        return enriched

    has_result_pronoun = bool(RESULT_PRONOUN_PATTERN.search(message))
    if has_results and entities.get("relationship") and has_result_pronoun:
        _inherit_filters(entities, filters)
        entities["result_ids"] = list(last_result_ids or [])
        enriched["intent"] = (
            previous_intent
            if previous_intent in CONTEXT_INTENTS
            else "birthday_search"
        )
        enriched["context_mode"] = "filter"
        enriched["context_used"] = True
        return enriched

    if has_results and entities.get("place") and has_reference:
        _inherit_filters(entities, filters)
        entities["result_ids"] = list(
            base_result_ids if base_result_ids is not None else last_result_ids
        )
        enriched["intent"] = (
            previous_intent
            if previous_intent in CONTEXT_INTENTS
            else "birthday_search"
        )
        enriched["context_mode"] = "filter"
        enriched["context_used"] = True
        return enriched

    if has_results and enriched.get("result_action"):
        entities["result_ids"] = list(last_result_ids or [])
        enriched["intent"] = (
            "result_count"
            if enriched["result_action"] == "count"
            else "result_list"
        )
        enriched["context_mode"] = "view"
        enriched["context_used"] = True
        return enriched

    if (
        has_results
        and enriched["intent"] == "count_birthdays"
        and re.search(r"\b(?:are there|matching people)\b", message, re.IGNORECASE)
    ):
        entities["result_ids"] = list(last_result_ids or [])
        enriched["intent"] = "result_count"
        enriched["context_mode"] = "view"
        enriched["context_used"] = True
        return enriched

    if (
        has_context
        and previous_intent in CONTEXT_INTENTS
        and entities.get("relationship")
        and (
            entities["relationship"] != filters.get("relationship")
            or re.search(r"\bwhat about\b", message, re.IGNORECASE)
        )
    ):
        _inherit_filters(entities, filters)
        enriched["intent"] = "birthday_search"
        enriched["context_mode"] = "base"
        enriched["context_used"] = True
        return enriched

    if has_results and has_result_pronoun:
        entities["result_ids"] = list(last_result_ids or [])
        _inherit_filters(entities, filters)
        if enriched["intent"] == "unknown" and previous_intent in CONTEXT_INTENTS:
            enriched["intent"] = previous_intent
        enriched["context_mode"] = "filter"
        enriched["context_used"] = True

    enriched["entities"] = entities
    return enriched


def update_state(
    parsed: dict[str, Any],
    state: dict[str, Any],
    last_result_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Update saved filters and visible/base result sets after a query."""
    updated = dict(state)
    entities = parsed.get("entities") or {}
    mode = parsed.get("context_mode", "base")
    filters = _state_filters(state)
    prior_result_ids = state.get("last_result_ids") or []
    incoming_person_id = entities.get("person_id")
    preserve_multi_result = (
        mode == "base"
        and len(prior_result_ids) > 1
        and incoming_person_id is not None
        and str(incoming_person_id) not in {str(value) for value in prior_result_ids}
        and not parsed.get("clarification_resolved")
    )
    if mode == "base" and not preserve_multi_result:
        filters = {
            key: entities[key]
            for key in FILTER_FIELDS
            if entities.get(key) is not None
        }
    elif mode == "filter":
        for key in FILTER_FIELDS:
            if entities.get(key) is not None:
                filters[key] = entities[key]

    for key in FILTER_FIELDS:
        updated[key] = filters.get(key)
    updated["previous_month"] = filters.get("month")
    updated["previous_relationship"] = filters.get("relationship")
    updated["previous_place"] = filters.get("place")
    updated["current_filters"] = filters

    if mode == "base" and not preserve_multi_result:
        previous_intent = parsed.get("intent")
        updated["previous_intent"] = previous_intent
        updated["last_intent"] = previous_intent
        updated["base_result_ids"] = (
            [str(value) for value in last_result_ids]
            if last_result_ids is not None
            else []
        )
        if "person_id" not in entities:
            updated["person_id"] = None
            updated["person_name"] = None

    if entities.get("person_id") is not None:
        updated["person_id"] = str(entities["person_id"])
        updated["previous_person_id"] = str(entities["person_id"])
        if entities.get("person_name"):
            updated["person_name"] = entities["person_name"]
    elif mode == "base":
        updated.setdefault("previous_person_id", state.get("person_id"))

    if preserve_multi_result:
        updated["last_result_ids"] = [str(value) for value in prior_result_ids]
        updated["previous_result_ids"] = [str(value) for value in prior_result_ids]
    elif last_result_ids is not None:
        result_ids = [str(person_id) for person_id in last_result_ids]
        updated["last_result_ids"] = result_ids
        updated["previous_result_ids"] = result_ids
    else:
        updated.setdefault("last_result_ids", [])
        updated.setdefault("previous_result_ids", updated["last_result_ids"])
    return updated


def resolve_clarification(
    message: str,
    state: dict[str, Any],
    people: pd.DataFrame,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a pending relationship question against only its candidate rows."""
    pending = state.get("pending_clarification")
    if not isinstance(pending, dict):
        return None, None

    candidate_ids = {str(value) for value in pending.get("candidate_ids", [])}
    if not candidate_ids or "Person_ID" not in people.columns:
        return None, "I couldn't find any people to choose from. Please ask the question again."

    candidates = people[
        people["Person_ID"].astype(str).isin(candidate_ids)
    ]
    candidate_parse = parse(message, candidates)
    entities = candidate_parse.get("entities") or {}
    person_id = entities.get("person_id")

    if person_id and str(person_id) in candidate_ids:
        resolved_entities = dict(entities)
        relationship = pending.get("relationship")
        if relationship:
            resolved_entities["relationship"] = relationship
        return {
            "intent": pending.get("intent", "birthday_search"),
            "entities": resolved_entities,
            "clarification_resolved": True,
        }, None

    relationship = pending.get("relationship", "person")
    names = ", ".join(candidates["Name"].astype(str).tolist())
    if candidate_parse.get("ambiguity"):
        return None, (
            f"I found multiple matching {relationship.lower()}s. "
            f"Please use a full name. Options: {names}."
        )
    return None, (
        f"I couldn't match that name among the {relationship.lower()}s. "
        f"Please reply with a name from: {names}."
    )

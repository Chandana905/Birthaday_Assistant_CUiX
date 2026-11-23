from __future__ import annotations

import calendar
import re
from typing import Any

import pandas as pd


MONTHS = {
    name.casefold(): month
    for month, name in enumerate(calendar.month_name)
    if name
}
COUNT_PATTERN = re.compile(r"\b(?:how many|count|number of|total)\b")
LOCATION_PATTERN = re.compile(r"\b(?:where|stay|stays|live|lives|location|place)\b")
UPCOMING_PATTERN = re.compile(
    r"\b(?:tomorrow|upcoming)\b|"
    r"\b(?:next|coming)\s+(?:7|30)\s+days\b"
)
BIRTHDAY_PATTERN = re.compile(
    r"\b(?:birthdays?|birth date|born|celebrat\w*|wish)\b"
)
PERSON_INFO_PATTERN = re.compile(r"\b(?:who\s+is|tell\s+me\s+about)\b")
PERSON_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:what|which)\s+(?:is|was)\s+(?:their|his|her)\s+relationship\b|"
    r"\brelationship\s+with\s+me\b"
)
PERSON_ID_PATTERN = re.compile(r"(?<!\w)P\d{3}(?!\w)", re.IGNORECASE)
RESULT_REFERENCE_PATTERN = re.compile(
    r"\b(?:they|them|their|those|these|there)\b", re.IGNORECASE
)
RESULT_SORT_PATTERN = re.compile(
    r"\b(?:sort|order)\b.*\b(?:birthday|birthdays|date)\b",
    re.IGNORECASE,
)
RESULT_LOCATION_PATTERN = re.compile(
    r"\b(?:show|list|display)\b.*\b(?:their|them)\b.*\blocations?\b",
    re.IGNORECASE,
)
RESULT_NAME_PATTERN = re.compile(
    r"\b(?:show|list|display)\b.*\b(?:their|them)\b.*\b(?:names?|birthdays?)\b",
    re.IGNORECASE,
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(
        r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)",
        text.casefold(),
    ) is not None


def _relationship_terms(people: pd.DataFrame) -> list[tuple[str, str]]:
    terms: dict[str, str] = {}
    if "Relationship" not in people.columns:
        return []

    for value in people["Relationship"].dropna().astype(str):
        label = value.strip()
        if not label:
            continue
        terms[label.casefold()] = label
        if not label.casefold().endswith("s"):
            plural = label + "s"
            terms[plural.casefold()] = label

    return sorted(terms.items(), key=lambda item: len(item[0]), reverse=True)


def _matching_place(people: pd.DataFrame, text: str) -> str | None:
    if "Place_of_Stay" not in people.columns:
        return None
    places = {
        str(value).strip()
        for value in people["Place_of_Stay"].dropna()
        if str(value).strip()
    }
    return next(
        (
            place
            for place in sorted(places, key=len, reverse=True)
            if _contains_phrase(text, place)
        ),
        None,
    )


def _matching_people(people: pd.DataFrame, text: str, column: str) -> set[Any]:
    if column not in people.columns:
        return set()
    matches: set[Any] = set()
    for index, value in people[column].items():
        if pd.isna(value):
            continue
        phrase = str(value).strip()
        if len(phrase) >= 3 and _contains_phrase(text, phrase):
            matches.add(index)
    return matches


def _person_entities(people: pd.DataFrame, text: str) -> tuple[dict[str, Any], bool]:
    if "Name" not in people.columns:
        return {}, False

    full_name_matches = _matching_people(people, text, "Name")
    nickname_matches = _matching_people(people, text, "Nickname")
    candidate_matches = full_name_matches or nickname_matches

    if len(full_name_matches) > 1:
        remaining_text = text
        for index in full_name_matches:
            name = str(people.loc[index, "Name"]).strip()
            remaining_text = re.sub(
                r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                " ",
                remaining_text,
                flags=re.IGNORECASE,
            )
        explicit_nicknames = _matching_people(
            people, remaining_text, "Nickname"
        )
        disambiguated = full_name_matches.intersection(explicit_nicknames)
        if len(disambiguated) == 1:
            candidate_matches = disambiguated
        else:
            return {}, True
    elif full_name_matches:
        candidate_matches = full_name_matches

    if len(candidate_matches) != 1:
        return {}, len(candidate_matches) > 1

    index = next(iter(candidate_matches))
    person = people.loc[index]
    return {
        "person_id": str(person["Person_ID"]) if "Person_ID" in people.columns else "",
        "person_name": str(person["Name"]).strip(),
    }, False


def parse(question: str, people: pd.DataFrame) -> dict[str, Any]:
    """Classify a single question and extract values grounded in the dataset."""
    text = re.sub(r"\s+", " ", question.casefold().strip()) if isinstance(question, str) else ""
    entities: dict[str, Any] = {}

    for month_name, month_number in MONTHS.items():
        if _contains_phrase(text, month_name):
            entities["month"] = month_number
            break

    if _contains_phrase(text, "tomorrow"):
        entities["relative_date"] = "tomorrow"
    if _contains_phrase(text, "next month"):
        entities["relative_month"] = "next month"

    for term, label in _relationship_terms(people):
        if _contains_phrase(text, term):
            entities["relationship"] = label
            break
    if "relationship" not in entities and "Relationship_Group" in people.columns:
        groups = sorted(
            {
                str(value).strip()
                for value in people["Relationship_Group"].dropna()
                if str(value).strip()
            },
            key=len,
            reverse=True,
        )
        for group in groups:
            if _contains_phrase(text, group):
                entities["relationship"] = group
                break
    place = _matching_place(people, text)
    if place:
        entities["place"] = place

    person_ids = PERSON_ID_PATTERN.findall(text)
    person_is_ambiguous = False
    if person_ids:
        if len(person_ids) == 1 and "Person_ID" in people.columns:
            id_matches = people[
                people["Person_ID"].astype(str).str.casefold()
                == person_ids[0].casefold()
            ]
            if len(id_matches) == 1:
                person = id_matches.iloc[0]
                entities["person_id"] = str(person["Person_ID"])
                entities["person_name"] = str(person["Name"]).strip()
    else:
        person_entities, person_is_ambiguous = _person_entities(people, text)
        entities.update(person_entities)

    result_action = None
    result_view = None
    sort_by = None
    person_detail = None
    if RESULT_SORT_PATTERN.search(text) and RESULT_REFERENCE_PATTERN.search(text):
        result_action = "list"
        result_view = "all"
        sort_by = "birthday"
    elif RESULT_LOCATION_PATTERN.search(text):
        result_action = "list"
        result_view = "locations"
    elif RESULT_NAME_PATTERN.search(text):
        result_action = "list"
        result_view = "names_birthdays"

    if PERSON_RELATIONSHIP_PATTERN.search(text):
        intent = "person_info"
        person_detail = "relationship"
    elif LOCATION_PATTERN.search(text):
        intent = "person_place"
    elif COUNT_PATTERN.search(text):
        intent = "count_birthdays"
    elif UPCOMING_PATTERN.search(text):
        intent = "upcoming_birthdays"
    elif BIRTHDAY_PATTERN.search(text):
        intent = "birthday_search"
    elif PERSON_INFO_PATTERN.search(text):
        intent = "person_info"
    else:
        intent = "unknown"

    parsed: dict[str, Any] = {"intent": intent, "entities": entities}
    if result_action:
        parsed["result_action"] = result_action
        parsed["result_view"] = result_view
    if sort_by:
        parsed["sort_by"] = sort_by
    if person_detail:
        parsed["person_detail"] = person_detail
    if person_is_ambiguous:
        parsed["ambiguity"] = ["person"]
    return parsed

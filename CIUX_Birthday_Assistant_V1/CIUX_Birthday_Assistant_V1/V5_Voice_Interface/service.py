from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _birthday_in_year(birthday: date, year: int) -> date:
    try:
        return date(year, birthday.month, birthday.day)
    except ValueError:
        if birthday.month == 2 and birthday.day == 29:
            return date(year, 2, 28)
        raise


def _next_birthday(birthday: date, today: date) -> date:
    occurrence = _birthday_in_year(birthday, today.year)
    if occurrence < today:
        occurrence = _birthday_in_year(birthday, today.year + 1)
    return occurrence


def format_rows(rows: pd.DataFrame) -> str:
    """Format matching spreadsheet rows as readable birthday lines."""
    if rows.empty:
        return "No matching people were found."

    lines = []
    for _, row in rows.iterrows():
        birthday = _as_date(row["DOB"])
        details = [f"{row['Name']} — {birthday.strftime('%d %B %Y')}"]
        relationship = row.get("Relationship")
        if pd.notna(relationship) and str(relationship).strip():
            details.append(f"relationship: {str(relationship).strip()}")
        place = row.get("Place_of_Stay")
        if pd.notna(place) and str(place).strip():
            details.append(f"location: {str(place).strip()}")
        lines.append("• " + "; ".join(details))
    return f"Found {len(rows)} matching people:\n" + "\n".join(lines)


def _filter_relationship_and_month(
    people: pd.DataFrame,
    entities: dict[str, Any],
    today: date,
) -> pd.DataFrame:
    rows = people
    relationship = entities.get("relationship")
    if relationship:
        if "Relationship" in people.columns and any(
            people["Relationship"].astype(str).str.casefold()
            == str(relationship).casefold()
        ):
            rows = rows[
                rows["Relationship"].astype(str).str.casefold()
                == str(relationship).casefold()
            ]
        elif "Relationship_Group" in people.columns:
            rows = rows[
                rows["Relationship_Group"].astype(str).str.casefold()
                == str(relationship).casefold()
            ]
        else:
            return people.iloc[0:0].copy()

    month = entities.get("month")
    if entities.get("relative_month") == "next month":
        month = (today.month % 12) + 1
    elif entities.get("relative_month") == "this month":
        month = today.month
    if month:
        rows = rows[pd.to_datetime(rows["DOB"]).dt.month == int(month)]
    place = entities.get("place")
    if place:
        rows = rows[
            rows["Place_of_Stay"].astype(str).str.casefold()
            == str(place).casefold()
        ]
    result_ids = entities.get("result_ids")
    if result_ids is not None and "Person_ID" in rows.columns:
        rows = rows[
            rows["Person_ID"].astype(str).isin({str(value) for value in result_ids})
        ]
    return rows.copy()


def find_result_ids(
    parsed: dict[str, Any],
    people: pd.DataFrame,
    today: date | None = None,
) -> list[str] | None:
    """Return workbook IDs corresponding to the parsed query, when applicable."""
    intent = parsed.get("intent")
    entities = parsed.get("entities") or {}
    if intent not in {
        "birthday_search",
        "count_birthdays",
        "upcoming_birthdays",
        "person_place",
        "person_info",
        "result_list",
        "result_count",
    }:
        if intent == "unknown" and entities.get("person_id"):
            matches = people[
                people["Person_ID"].astype(str).str.casefold()
                == str(entities["person_id"]).casefold()
            ]
            return matches["Person_ID"].astype(str).tolist()
        return None

    today = today or date.today()
    rows = _filter_relationship_and_month(people, entities, today)
    person_id = entities.get("person_id")
    if person_id is not None and "Person_ID" in people.columns:
        person_rows = people[
            people["Person_ID"].astype(str).str.casefold()
            == str(person_id).casefold()
        ]
        rows = rows[
            rows["Person_ID"].astype(str).isin(
                person_rows["Person_ID"].astype(str)
            )
        ]
    elif entities.get("relative_date") in {"today", "tomorrow"}:
        offset = 0 if entities["relative_date"] == "today" else 1
        rows = _on_date(rows, today + timedelta(days=offset))
    elif intent == "upcoming_birthdays":
        rows = _upcoming(rows, today)

    if "Person_ID" not in rows.columns:
        return None
    return rows["Person_ID"].astype(str).tolist()


def _on_date(rows: pd.DataFrame, target: date) -> pd.DataFrame:
    matching_indices = []
    for index, row in rows.iterrows():
        birthday = _as_date(row["DOB"])
        if _birthday_in_year(birthday, target.year) == target:
            matching_indices.append(index)
    return rows.loc[matching_indices].copy()


def _upcoming(rows: pd.DataFrame, today: date, days: int = 30) -> pd.DataFrame:
    end = today + timedelta(days=days)
    upcoming = rows.copy()
    upcoming["_next_birthday"] = [
        _next_birthday(_as_date(value), today)
        for value in upcoming["DOB"]
    ]
    upcoming = upcoming[
        (upcoming["_next_birthday"] >= today)
        & (upcoming["_next_birthday"] <= end)
    ]
    return upcoming.sort_values(
        ["_next_birthday", "Name"], kind="stable"
    )


def _format_upcoming(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No matching birthdays were found in the next 30 days."

    lines = []
    for _, row in rows.iterrows():
        occurrence = _as_date(row["_next_birthday"])
        details = [f"{row['Name']} — {occurrence.strftime('%d %B %Y')}"]
        relationship = row.get("Relationship")
        if pd.notna(relationship) and str(relationship).strip():
            details.append(f"relationship: {str(relationship).strip()}")
        place = row.get("Place_of_Stay")
        if pd.notna(place) and str(place).strip():
            details.append(f"location: {str(place).strip()}")
        lines.append("• " + "; ".join(details))
    return f"Found {len(rows)} matching people:\n" + "\n".join(lines)


def execute(
    parsed: dict[str, Any],
    people: pd.DataFrame,
    today: date | None = None,
) -> str:
    """Query the Excel-backed DataFrame and return a user-facing answer."""
    today = today or date.today()
    intent = parsed.get("intent", "unknown")
    entities = parsed.get("entities") or {}
    rows = _filter_relationship_and_month(people, entities, today)
    person_id = entities.get("person_id")
    if person_id is not None and "Person_ID" in people.columns:
        person_rows = people[
            people["Person_ID"].astype(str).str.casefold()
            == str(person_id).casefold()
        ]
        rows = rows[
            rows["Person_ID"].astype(str).isin(
                person_rows["Person_ID"].astype(str)
            )
        ]

    if intent == "unknown":
        person_id = entities.get("person_id")
        if person_id:
            matches = people[
                people["Person_ID"].astype(str).str.casefold()
                == str(person_id).casefold()
            ]
            if len(matches) == 1:
                person_name = str(matches.iloc[0]["Name"])
                return (
                    f"What would you like to know about {person_name}? "
                    "I can look up their birthday or where they stay."
                )
        return "I couldn't understand that question. Please rephrase it as a birthday, count, upcoming-birthday, or person-location question."

    if intent == "person_info":
        person_id = entities.get("person_id")
        if not person_id:
            return "Please include a person's name or Person_ID so I can look them up."
        matches = people[
            people["Person_ID"].astype(str).str.casefold()
            == str(person_id).casefold()
        ]
        if matches.empty:
            return "I couldn't find that person in the spreadsheet."
        person = matches.iloc[0]
        if parsed.get("person_detail") == "relationship":
            relationship = person.get("Relationship")
            if pd.isna(relationship) or not str(relationship).strip():
                return f"No relationship is recorded for {person['Name']}."
            return f"{person['Name']}'s relationship is {str(relationship).strip()}."
        details = []
        relationship = person.get("Relationship")
        if pd.notna(relationship) and str(relationship).strip():
            details.append(f"relationship: {str(relationship).strip()}")
        if "DOB" in person and pd.notna(person["DOB"]):
            birthday = _as_date(person["DOB"])
            details.append(f"birthday: {birthday.strftime('%d %B %Y')}")
        place = person.get("Place_of_Stay")
        if pd.notna(place) and str(place).strip():
            details.append(f"stays in: {str(place).strip()}")
        name = str(person["Name"]).strip()
        return f"{name} — " + "; ".join(details) + "."

    if intent == "result_count":
        return f"There are {len(rows)} matching people."

    if intent == "result_list":
        if rows.empty:
            return "No matching people were found."
        if parsed.get("sort_by") == "birthday":
            rows = rows.assign(
                _birthday_sort=pd.to_datetime(rows["DOB"]).dt.strftime("%m-%d")
            ).sort_values(["_birthday_sort", "Name"], kind="stable")
        view = parsed.get("result_view", "all")
        if view == "names":
            details = [f"• {name}" for name in rows["Name"].astype(str)]
            return f"Found {len(rows)} matching people:\n" + "\n".join(details)
        if view == "locations":
            details = [
                f"• {row['Name']} — "
                f"{row['Place_of_Stay'] if pd.notna(row.get('Place_of_Stay')) else 'location not recorded'}"
                for _, row in rows.iterrows()
            ]
            return f"Found {len(rows)} matching people:\n" + "\n".join(details)
        return format_rows(rows)

    if intent == "person_place":
        if "person" in parsed.get("ambiguity", []):
            return "I found multiple people matching that name. Please include a unique nickname or another identifying detail."
        person_id = entities.get("person_id")
        if not person_id:
            place = entities.get("place")
            if place:
                matching = people[
                    people["Place_of_Stay"].astype(str).str.casefold()
                    == str(place).casefold()
                ]
                if matching.empty:
                    return f"I couldn't find anyone recorded in {place}."
                return "People staying in " + place + ":\n" + "\n".join(
                    f"• {name}" for name in matching["Name"].astype(str)
                )
            return "Please include the person's name so I can look up their location."
        matches = people[
            people["Person_ID"].astype(str).str.casefold()
            == str(person_id).casefold()
        ]
        if matches.empty:
            return "I couldn't find that person in the spreadsheet."
        person = matches.iloc[0]
        return f"{person['Name']} stays in {person['Place_of_Stay']}."

    if intent == "count_birthdays":
        if entities.get("relative_date") in {"today", "tomorrow"}:
            offset = 0 if entities["relative_date"] == "today" else 1
            rows = _on_date(rows, today + timedelta(days=offset))
        return f"I found {len(rows)} matching birthdays."

    if intent == "upcoming_birthdays":
        if entities.get("relative_date") in {"today", "tomorrow"}:
            offset = 0 if entities["relative_date"] == "today" else 1
            label = entities["relative_date"]
            rows = _on_date(rows, today + timedelta(days=offset))
            if rows.empty:
                return f"No matching birthdays were found {label}."
            return f"Birthdays {label}:\n" + format_rows(rows)

        upcoming = _upcoming(rows, today)
        if upcoming.empty:
            return "No matching birthdays were found in the next 30 days."
        return "Upcoming birthdays in the next 30 days:\n" + _format_upcoming(upcoming)

    if intent == "birthday_search":
        if "person" in parsed.get("ambiguity", []):
            return "I found multiple people matching that name. Please include a unique nickname or another identifying detail."

        person_id = entities.get("person_id")
        if person_id:
            matches = people[
                people["Person_ID"].astype(str).str.casefold()
                == str(person_id).casefold()
            ]
            if matches.empty:
                return "I couldn't find that person in the spreadsheet."
            person = matches.iloc[0]
            birthday = _as_date(person["DOB"])
            return (
                f"{person['Name']}'s birthday is "
                f"{birthday.strftime('%d %B %Y')}."
            )

        if entities.get("relative_date") in {"today", "tomorrow"}:
            offset = 0 if entities["relative_date"] == "today" else 1
            rows = _on_date(rows, today + timedelta(days=offset))
        if rows.empty:
            if entities.get("month") or entities.get("relative_month"):
                return "No matching birthdays were found for that time period."
            if entities.get("relationship"):
                return (
                    "No matching birthdays were found for "
                    f"{entities['relationship']}."
                )
            return "Please include a person's name, relationship, or month."

        if len(rows) == 1:
            person = rows.iloc[0]
            birthday = _as_date(person["DOB"])
            return (
                f"{person['Name']}'s birthday is "
                f"{birthday.strftime('%d %B %Y')}."
            )
        return format_rows(rows)

    return "I couldn't understand that question. Please rephrase it."

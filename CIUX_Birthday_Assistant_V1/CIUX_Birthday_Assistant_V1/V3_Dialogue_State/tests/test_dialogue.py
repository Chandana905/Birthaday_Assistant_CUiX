from __future__ import annotations

import pandas as pd
import pytest

from app import app, PEOPLE
from dialogue import enrich_with_context, resolve_clarification, update_state
from nlu import parse
from service import execute


def nephew_september_ids():
    matches = PEOPLE[
        (PEOPLE["Relationship"].astype(str).str.casefold() == "nephew")
        & (pd.to_datetime(PEOPLE["DOB"]).dt.month == 9)
    ]
    return set(matches["Person_ID"].astype(str))


def ask(client, message):
    response = client.post("/api/chat", json={"message": message})
    assert response.status_code == 200
    return response.get_json()


def test_context_carries_missing_relationship_into_new_month():
    state = {
        "month": 9,
        "relationship": "Nephew",
        "last_intent": "birthday_search",
    }
    parsed = enrich_with_context(
        parse("What about October?", PEOPLE),
        state,
    )
    assert parsed["intent"] == "birthday_search"
    assert parsed["entities"]["month"] == 10
    assert parsed["entities"]["relationship"] == "Nephew"


def test_context_carries_missing_month_into_new_relationship():
    state = {
        "month": 9,
        "relationship": "Nephew",
        "last_intent": "birthday_search",
    }
    parsed = enrich_with_context(parse("What about nieces?", PEOPLE), state)
    assert parsed["intent"] == "birthday_search"
    assert parsed["entities"]["month"] == 9
    assert parsed["entities"]["relationship"] == "Niece"


def test_new_entities_replace_old_context():
    state = {
        "month": 9,
        "relationship": "Nephew",
        "last_intent": "birthday_search",
    }
    parsed = enrich_with_context(
        parse("Any niece birthdays in October?", PEOPLE),
        state,
    )
    updated = update_state(parsed, state, ["P100"])
    assert updated["month"] == 10
    assert updated["relationship"] == "Niece"
    assert updated["last_result_ids"] == ["P100"]


def test_update_state_records_person_details():
    parsed = {
        "intent": "birthday_search",
        "entities": {"person_id": "P006", "person_name": "Nikhil Hegde"},
    }
    updated = update_state(parsed, {}, ["P006"])
    assert updated["person_id"] == "P006"
    assert updated["person_name"] == "Nikhil Hegde"
    assert updated["last_result_ids"] == ["P006"]


def test_nikhil_resolves_within_nephew_clarification_candidates():
    candidates = PEOPLE[PEOPLE["Relationship"] == "Nephew"]
    state = {
        "pending_clarification": {
            "relationship": "Nephew",
            "intent": "birthday_search",
            "candidate_ids": candidates["Person_ID"].astype(str).tolist(),
        }
    }
    parsed, message = resolve_clarification("Nikhil", state, PEOPLE)
    assert message is None
    assert parsed is not None
    assert parsed["entities"]["person_id"] == "P006"
    assert parsed["entities"]["person_name"] == "Nikhil Hegde"


def test_unmatched_clarification_name_is_reported():
    candidates = PEOPLE[PEOPLE["Relationship"] == "Nephew"]
    state = {
        "pending_clarification": {
            "relationship": "Nephew",
            "intent": "birthday_search",
            "candidate_ids": candidates["Person_ID"].astype(str).tolist(),
        }
    }
    parsed, message = resolve_clarification("Someone Unknown", state, PEOPLE)
    assert parsed is None
    assert "couldn't match" in message.lower()


def test_v2_birthday_search_remains_available():
    parsed = parse("When is Nikhil Hegde's birthday?", PEOPLE)
    answer = execute(parsed, PEOPLE)
    assert parsed["intent"] == "birthday_search"
    assert "Nikhil Hegde" in answer
    assert PEOPLE.loc[PEOPLE["Person_ID"] == "P006", "DOB"].iloc[0].strftime(
        "%d %B %Y"
    ) in answer


def test_flask_relationship_clarification_then_nikhil():
    client = app.test_client()
    first = client.post(
        "/api/chat",
        json={"message": "When is my nephew's birthday?"},
    ).get_json()
    assert "Which one do you mean?" in first["answer"]
    assert first["dialogue_state"]["pending_clarification"]["relationship"] == "Nephew"
    assert len(first["dialogue_state"]["last_result_ids"]) > 1

    answer = client.post("/api/chat", json={"message": "Nikhil"}).get_json()
    person = PEOPLE[PEOPLE["Person_ID"] == "P006"].iloc[0]
    assert person["Name"] in answer["answer"]
    assert person["DOB"].strftime("%d %B %Y") in answer["answer"]
    assert answer["debug"]["entities"]["person_id"] == "P006"
    assert "pending_clarification" not in answer["dialogue_state"]


def test_flask_month_follow_up_keeps_relationship_and_changes_month():
    client = app.test_client()
    first = client.post(
        "/api/chat",
        json={"message": "Any nephew birthdays in September?"},
    ).get_json()
    assert first["debug"]["entities"]["relationship"] == "Nephew"
    second = client.post(
        "/api/chat",
        json={"message": "What about October?"},
    ).get_json()
    assert second["debug"]["intent"] == "birthday_search"
    assert second["debug"]["entities"]["month"] == 10
    assert second["debug"]["entities"]["relationship"] == "Nephew"
    assert second["dialogue_state"]["month"] == 10


def test_flask_relationship_follow_up_keeps_month():
    client = app.test_client()
    client.post("/api/chat", json={"message": "Birthdays in September."})
    second = client.post(
        "/api/chat",
        json={"message": "What about nieces?"},
    ).get_json()
    assert second["debug"]["entities"]["month"] == 9
    assert second["debug"]["entities"]["relationship"] == "Niece"


def test_reset_endpoint_clears_session_and_context():
    client = app.test_client()
    client.post(
        "/api/chat",
        json={"message": "Any nephew birthdays in September?"},
    )
    reset_response = client.post("/reset").get_json()
    assert reset_response["success"] is True
    with client.session_transaction() as browser_session:
        assert "dialogue_state" not in browser_session

    follow_up = client.post(
        "/api/chat",
        json={"message": "What about October?"},
    ).get_json()
    assert follow_up["debug"]["intent"] == "unknown"
    assert follow_up["dialogue_state"].get("relationship") is None
    assert not follow_up["debug"].get("context_used", False)


def test_previous_nephew_september_results_survive_count_and_follow_ups():
    client = app.test_client()
    expected_ids = nephew_september_ids()
    assert len(expected_ids) == 3

    initial = ask(client, "Show all nephew birthdays in September.")
    assert set(initial["dialogue_state"]["last_result_ids"]) == expected_ids

    count = ask(client, "How many are there?")
    assert count["answer"] == f"I found {len(expected_ids)} matching birthdays."
    assert set(count["dialogue_state"]["last_result_ids"]) == expected_ids

    names = ask(client, "Show their names.")
    assert set(names["dialogue_state"]["last_result_ids"]) == expected_ids
    for person_id in expected_ids:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in names["answer"]

    locations = ask(client, "Where do they stay?")
    assert set(locations["dialogue_state"]["last_result_ids"]) == expected_ids
    for person_id in expected_ids:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in locations["answer"]
        assert person["Place_of_Stay"] in locations["answer"]

    birthdays = ask(client, "Tell me their birthdays.")
    assert set(birthdays["dialogue_state"]["last_result_ids"]) == expected_ids
    for person_id in expected_ids:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in birthdays["answer"]
        assert person["DOB"].strftime("%d %B %Y") in birthdays["answer"]


@pytest.mark.parametrize("message", ["List them.", "Who are they?"])
def test_name_follow_ups_use_only_previous_result_ids(message):
    client = app.test_client()
    expected_ids = nephew_september_ids()
    ask(client, "Show all nephew birthdays in September.")

    result = ask(client, message)

    assert result["debug"]["intent"] == "result_list"
    assert set(result["dialogue_state"]["last_result_ids"]) == expected_ids
    for person_id in expected_ids:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in result["answer"]


def test_reset_clears_all_filters_and_prevents_reusing_previous_results():
    client = app.test_client()
    ask(client, "Show all nephew birthdays in September.")

    reset = client.post("/reset").get_json()
    assert reset["success"] is True
    state = reset["dialogue_state"]
    assert state["month"] is None
    assert state["relationship"] is None
    assert state["last_result_ids"] == []
    assert state["person_id"] is None
    assert state["person_name"] is None

    follow_up = ask(client, "Show their names.")
    assert follow_up["dialogue_state"]["last_result_ids"] == []
    assert not follow_up["debug"].get("context_used", False)
    assert "birthday question first" in follow_up["answer"].lower()


def test_unresolved_person_location_does_not_store_the_full_dataset_as_results():
    client = app.test_client()

    response = ask(client, "Where does Sam stay?")

    assert "include the person's name" in response["answer"].lower()
    assert response["dialogue_state"]["last_result_ids"] == []


def test_reset_command_in_chat_clears_context():
    client = app.test_client()
    client.post(
        "/api/chat",
        json={"message": "Any nephew birthdays in September?"},
    )
    reset_response = client.post(
        "/api/chat",
        json={"message": "Reset the conversation."},
    ).get_json()
    assert reset_response["debug"]["intent"] == "reset"
    with client.session_transaction() as browser_session:
        assert "dialogue_state" not in browser_session


def test_home_health_and_reset_routes():
    client = app.test_client()
    assert client.get("/").status_code == 200
    health = client.get("/api/health").get_json()
    assert health["status"] == "ok"
    assert health["version"] == "V3"
    assert client.post("/reset").status_code == 200


def test_context_uses_only_entities_from_explicit_follow_up():
    people = pd.DataFrame(
        [
            {
                "Person_ID": "T001",
                "Name": "Test Person",
                "Nickname": "Tester",
                "DOB": pd.Timestamp("2000-09-10"),
                "Place_of_Stay": "Test City",
                "Relationship": "Niece",
                "Relationship_Group": "Family",
                "Relationship_Detail": "Niece",
            }
        ]
    )
    state = {
        "month": 9,
        "relationship": "Nephew",
        "last_intent": "birthday_search",
    }
    parsed = enrich_with_context(parse("What about October?", people), state)
    assert parsed["entities"]["month"] == 10
    assert parsed["entities"]["relationship"] == "Nephew"

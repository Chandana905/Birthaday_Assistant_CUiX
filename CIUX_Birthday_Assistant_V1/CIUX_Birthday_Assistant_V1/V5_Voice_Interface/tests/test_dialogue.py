from __future__ import annotations

import pandas as pd

import app as app_module
from app import app, PEOPLE
from dialogue import enrich_with_context, resolve_clarification, update_state
from llm_nlu import LLMNLUError
from nlu import parse
from service import execute, find_result_ids


def _ask(client, message):
    response = client.post("/api/chat", json={"message": message})
    assert response.status_code == 200
    return response.get_json()


def _ids_for(*, month=None, relationship=None, place=None):
    rows = PEOPLE
    if month is not None:
        rows = rows[pd.to_datetime(rows["DOB"]).dt.month == month]
    if relationship is not None:
        rows = rows[
            rows["Relationship"].astype(str).str.casefold()
            == relationship.casefold()
        ]
    if place is not None:
        rows = rows[
            rows["Place_of_Stay"].astype(str).str.casefold()
            == place.casefold()
        ]
    return set(rows["Person_ID"].astype(str))


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
    assert reset_response["dialogue_state"]["current_filters"] == {}
    assert reset_response["dialogue_state"]["base_result_ids"] == []
    with client.session_transaction() as browser_session:
        assert "dialogue_state" not in browser_session

    follow_up = client.post(
        "/api/chat",
        json={"message": "What about October?"},
    ).get_json()
    assert follow_up["debug"]["intent"] == "unknown"
    assert follow_up["dialogue_state"].get("relationship") is None
    assert not follow_up["debug"].get("context_used", False)


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
    assert health["version"] == "V4"
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


def test_conversation_a_preserves_month_relationship_and_result_filters():
    client = app.test_client()
    september_nephews = _ask(client, "Show me all nephew birthdays in September.")
    assert set(september_nephews["dialogue_state"]["last_result_ids"]) == _ids_for(
        month=9, relationship="Nephew"
    )

    october_nephews = _ask(client, "What about October?")
    expected_october = _ids_for(month=10, relationship="Nephew")
    assert set(october_nephews["dialogue_state"]["last_result_ids"]) == expected_october
    for person_id in expected_october:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in october_nephews["answer"]
        assert person["DOB"].strftime("%d %B %Y") in october_nephews["answer"]

    bengaluru = _ask(client, "Only those who stay in Bengaluru.")
    expected_bengaluru = _ids_for(
        month=10, relationship="Nephew", place="Bengaluru"
    )
    assert set(bengaluru["dialogue_state"]["last_result_ids"]) == expected_bengaluru

    count = _ask(client, "How many are there?")
    assert count["answer"] == f"There are {len(expected_bengaluru)} matching people."
    listed = _ask(client, "Show their names and birthdays.")
    assert set(listed["dialogue_state"]["last_result_ids"]) == expected_bengaluru
    for person_id in expected_bengaluru:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in listed["answer"]
        assert person["DOB"].strftime("%d %B %Y") in listed["answer"]


def test_conversation_b_replaces_relationship_then_filters_and_views_places():
    client = app.test_client()
    _ask(client, "Show birthdays in September.")
    nieces = _ask(client, "What about my nieces?")
    september_nieces = _ids_for(month=9, relationship="Niece")
    assert set(nieces["dialogue_state"]["last_result_ids"]) == september_nieces

    mysuru = _ask(client, "Only those in Mysuru.")
    expected_mysuru = _ids_for(
        month=9, relationship="Niece", place="Mysuru"
    )
    assert set(mysuru["dialogue_state"]["last_result_ids"]) == expected_mysuru

    locations = _ask(client, "Show their locations.")
    assert set(locations["dialogue_state"]["last_result_ids"]) == expected_mysuru
    for person_id in expected_mysuru:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in locations["answer"]
        assert person["Place_of_Stay"] in locations["answer"]


def test_place_question_and_birthday_sort_filter_the_previous_result_set():
    location_client = app.test_client()
    initial = _ask(location_client, "Show birthdays in September.")

    mysuru = _ask(location_client, "What about Mysuru?")
    expected_mysuru = _ids_for(month=9, place="Mysuru")
    assert set(mysuru["dialogue_state"]["last_result_ids"]) == expected_mysuru

    sort_client = app.test_client()
    initial = _ask(sort_client, "Show birthdays in September.")
    september_ids = set(initial["dialogue_state"]["last_result_ids"])
    sorted_result = _ask(sort_client, "Sort them by birthday.")
    expected_order = (
        PEOPLE[
            PEOPLE["Person_ID"].astype(str).isin(september_ids)
        ]
        .assign(_sort=pd.to_datetime(
            PEOPLE.loc[
                PEOPLE["Person_ID"].astype(str).isin(september_ids), "DOB"
            ]
        ).dt.strftime("%m-%d"))
        .sort_values(["_sort", "Name"], kind="stable")["Name"]
        .astype(str)
        .tolist()
    )
    positions = [sorted_result["answer"].index(name) for name in expected_order]
    assert positions == sorted(positions)


def test_conversation_c_filters_upcoming_set_then_replaces_month():
    client = app.test_client()
    upcoming = _ask(client, "Show all upcoming birthdays.")
    all_upcoming_ids = set(find_result_ids(parse("Show all upcoming birthdays.", PEOPLE), PEOPLE))
    assert set(upcoming["dialogue_state"]["last_result_ids"]) == all_upcoming_ids

    nephews = _ask(client, "Which of them are nephews?")
    expected_upcoming_nephews = all_upcoming_ids & _ids_for(relationship="Nephew")
    assert set(nephews["dialogue_state"]["last_result_ids"]) == expected_upcoming_nephews

    october = _ask(client, "What about October?")
    expected_october = _ids_for(month=10, relationship="Nephew")
    assert set(october["dialogue_state"]["last_result_ids"]) == expected_october
    count = _ask(client, "How many matching people are there?")
    assert count["answer"] == f"There are {len(expected_october)} matching people."
    assert set(count["dialogue_state"]["last_result_ids"]) == expected_october


def test_conversation_d_resolves_person_pronouns_from_workbook():
    client = app.test_client()
    birthday = _ask(client, "Show birthday of P001.")
    person = PEOPLE[PEOPLE["Person_ID"].astype(str) == "P001"].iloc[0]
    assert person["Name"] in birthday["answer"]
    assert person["DOB"].strftime("%d %B %Y") in birthday["answer"]

    location = _ask(client, "Where does this person stay?")
    assert person["Place_of_Stay"] in location["answer"]
    assert location["debug"]["entities"]["person_id"] == "P001"

    relationship = _ask(client, "What is their relationship with me?")
    assert str(person["Relationship"]) in relationship["answer"]
    assert relationship["debug"]["entities"]["person_id"] == "P001"


def test_unrelated_person_lookup_does_not_replace_multi_person_context():
    client = app.test_client()
    initial = _ask(client, "Show me all nephew birthdays in September.")
    original_ids = set(initial["dialogue_state"]["last_result_ids"])
    assert len(original_ids) > 1

    unrelated = PEOPLE[
        ~PEOPLE["Person_ID"].astype(str).isin(original_ids)
    ].iloc[0]
    answer = _ask(client, f"Show birthday of {unrelated['Person_ID']}.")
    assert unrelated["Name"] in answer["answer"]
    assert set(answer["dialogue_state"]["last_result_ids"]) == original_ids

    listing = _ask(client, "Show their names and birthdays.")
    assert set(listing["dialogue_state"]["last_result_ids"]) == original_ids
    for person_id in original_ids:
        person = PEOPLE[PEOPLE["Person_ID"].astype(str) == person_id].iloc[0]
        assert person["Name"] in listing["answer"]


def test_groq_failure_logs_safe_category_and_keeps_rule_fallback(
    monkeypatch, caplog
):
    monkeypatch.setattr(app_module, "get_configured_api_key", lambda: "test-key")
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        lambda _message: (_ for _ in ()).throw(
            LLMNLUError(
                "Groq NLU network failure",
                category="network",
                diagnostic="Could not connect; check network access.",
            )
        ),
    )
    result = _ask(app.test_client(), "Show upcoming birthdays.")

    assert result["debug"]["nlu_method"] == "Rule-based fallback"
    assert "network" in result["debug"]["fallback_reason"]
    assert "Upcoming birthdays" in result["answer"]
    assert "category=network" in caplog.text
    assert "test-key" not in caplog.text

from __future__ import annotations

import re
from unittest.mock import Mock

import pytest

import app as app_module
from app import PEOPLE, app, validate_and_ground
from llm_nlu import LLMNLUError


def llm_result(intent, entities):
    return {"intent": intent, "entities": entities}


def test_month_and_relationship_are_validated_against_allowed_values():
    parsed, status, _ = validate_and_ground(
        llm_result(
            "birthday_search",
            {"relationship": "nephew", "month": 9},
        ),
        "Do I need to wish any nephew during September?",
    )
    assert parsed["entities"]["relationship"] == "Nephew"
    assert parsed["entities"]["month"] == 9
    assert "Validated" in status


def test_sibling_family_is_grounded_as_workbook_group():
    parsed, _, _ = validate_and_ground(
        llm_result(
            "birthday_search",
            {"relationship": "Sibling Family", "relative_month": "next month"},
        ),
        "Are there people from my sibling family celebrating next month?",
    )
    assert parsed["entities"]["relationship"] == "Sibling Family"
    assert parsed["entities"]["relative_month"] == "next month"


@pytest.mark.parametrize(
    "intent,entities",
    [
        ("made_up", {}),
        ("birthday_search", {"month": 13}),
        ("birthday_search", {"relationship": "Imaginary Relative"}),
        ("person_place", {"place": "Imaginary City"}),
        ("birthday_search", {"date_of_birth": "2020-01-01"}),
        ("birthday_search", {"relative_date": "last year"}),
    ],
)
def test_invalid_intents_entities_and_values_are_rejected(intent, entities):
    with pytest.raises(ValueError):
        validate_and_ground(llm_result(intent, entities), "question")


def test_person_name_is_mapped_to_workbook_person_id():
    parsed, status, _ = validate_and_ground(
        llm_result("upcoming_birthdays", {"person_name": "Rahul"}),
        "I was wondering whether Rahul has a birthday coming soon.",
    )
    person = PEOPLE[PEOPLE["Person_ID"] == parsed["entities"]["person_id"]].iloc[0]
    assert person["Name"] == "Rahul Patil"
    assert parsed["entities"]["person_name"] == "Rahul Patil"
    assert "Person_ID" in status


def test_model_supplied_person_id_is_never_used():
    parsed, status, _ = validate_and_ground(
        llm_result(
            "birthday_search",
            {"person_name": "Rahul", "person_id": "P999"},
        ),
        "When is Rahul's birthday?",
    )
    assert parsed["entities"]["person_id"] != "P999"
    assert parsed["entities"]["person_id"] == "P004"
    assert "Person_ID" in status


def test_unknown_person_is_not_assigned_an_id_or_queried():
    client = app.test_client()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
        monkeypatch.setattr(
            app_module,
            "parse_with_llm",
            lambda message: llm_result(
                "birthday_search", {"person_name": "Unknown Person"}
            ),
        )
        response = client.post(
            "/api/chat",
            json={"message": "When is Unknown Person's birthday?"},
        ).get_json()
    assert "couldn't find that person" in response["answer"].lower()
    assert "person_id" not in response["debug"]["entities"]
    assert "no query executed" in response["debug"]["grounding_status"].lower()


def test_ambiguous_person_is_not_selected():
    duplicate_nickname = (
        PEOPLE["Nickname"].fillna("").astype(str).str.strip()
    )
    counts = duplicate_nickname[duplicate_nickname != ""].value_counts()
    nickname = counts[counts > 1].index[0]
    parsed, status, candidates = validate_and_ground(
        llm_result("birthday_search", {"person_name": nickname}),
        f"When is {nickname}'s birthday?",
    )
    assert parsed["ambiguity"] == ["person"]
    assert "person_id" not in parsed["entities"]
    assert len(candidates) > 1
    assert "ambiguous" in status.lower()


def test_llm_cannot_choose_one_person_from_ambiguous_nickname():
    parsed, status, candidates = validate_and_ground(
        llm_result("birthday_search", {"person_name": "Nikhil Hegde"}),
        "When is Nikhil's birthday?",
    )
    assert parsed["ambiguity"] == ["person"]
    assert "person_id" not in parsed["entities"]
    assert len(candidates) > 1
    assert "ambiguous" in status.lower()


def test_llm_api_failure_uses_rule_fallback(monkeypatch):
    client = app.test_client()
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        Mock(side_effect=TimeoutError("simulated timeout")),
    )
    response = client.post(
        "/api/chat",
        json={"message": "When is Rahul Patil's birthday?"},
    ).get_json()
    assert response["debug"]["nlu_method"] == "Rule-based fallback"
    assert "request failed" in response["trace"][2].lower() or any(
        "fallback reason" in item.lower() for item in response["trace"]
    )
    assert "Rahul Patil" in response["answer"]


def test_no_api_key_uses_rule_fallback(monkeypatch):
    client = app.test_client()
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "parse_with_llm", Mock(side_effect=AssertionError))
    response = client.post(
        "/api/chat",
        json={"message": "When is Rahul Patil's birthday?"},
    ).get_json()
    assert response["debug"]["nlu_method"] == "Rule-based fallback"
    assert "not configured" in response["debug"]["fallback_reason"]
    assert "Rahul Patil" in response["answer"]


def test_invalid_llm_output_uses_rule_fallback(monkeypatch):
    client = app.test_client()
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        Mock(side_effect=LLMNLUError("The LLM response was not valid JSON.")),
    )
    response = client.post(
        "/api/chat",
        json={"message": "When is Rahul Patil's birthday?"},
    ).get_json()
    assert response["debug"]["nlu_method"] == "Rule-based fallback"
    assert "not valid json" in response["debug"]["fallback_reason"].lower()
    assert "Rahul Patil" in response["answer"]


def test_unknown_name_in_rule_fallback_does_not_query_entire_dataset(monkeypatch):
    client = app.test_client()
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post(
        "/api/chat",
        json={"message": "When is Unknown Person's birthday?"},
    ).get_json()
    assert "couldn't find that person" in response["answer"].lower()
    assert "no query executed" in response["debug"]["grounding_status"].lower()


def test_ambiguous_llm_person_is_clarified_and_resolved_from_candidates(monkeypatch):
    client = app.test_client()
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    nickname = (
        PEOPLE["Nickname"].fillna("").astype(str).str.strip()
        .value_counts()
    )
    duplicate_nickname = nickname[nickname.index != ""]
    duplicate_nickname = duplicate_nickname[duplicate_nickname > 1].index[0]
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        lambda message: llm_result(
            "birthday_search", {"person_name": duplicate_nickname}
        ),
    )
    first = client.post(
        "/api/chat",
        json={"message": f"When is {duplicate_nickname}'s birthday?"},
    ).get_json()
    assert "Which one do you mean?" in first["answer"]
    candidate_id = first["dialogue_state"]["pending_clarification"][
        "candidate_ids"
    ][0]
    expected_person = PEOPLE[
        PEOPLE["Person_ID"].astype(str) == candidate_id
    ].iloc[0]
    second = client.post(
        "/api/chat",
        json={"message": expected_person["Name"]},
    ).get_json()
    assert expected_person["Name"] in second["answer"]
    assert expected_person["DOB"].strftime("%d %B %Y") in second["answer"]


def test_llm_checkpoint_questions_keep_results_grounded(monkeypatch):
    client = app.test_client()
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    outputs = {
        "Do I need to wish any nephew during September?": llm_result(
            "birthday_search", {"relationship": "Nephew", "month": 9}
        ),
        "Are there people from my sibling family celebrating next month?": llm_result(
            "birthday_search",
            {"relationship": "Sibling Family", "relative_month": "next month"},
        ),
        "I was wondering whether Rahul has a birthday coming soon.": llm_result(
            "upcoming_birthdays", {"person_name": "Rahul"}
        ),
    }
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        lambda message: outputs[message],
    )
    for message, expected_intent in (
        ("Do I need to wish any nephew during September?", "birthday_search"),
        (
            "Are there people from my sibling family celebrating next month?",
            "birthday_search",
        ),
        (
            "I was wondering whether Rahul has a birthday coming soon.",
            "upcoming_birthdays",
        ),
    ):
        response = client.post(
            "/api/chat", json={"message": message}
        ).get_json()
        assert response["debug"]["intent"] == expected_intent
        assert response["debug"]["nlu_method"] == "LLM NLU"
        assert response["debug"]["grounding_status"].startswith("Validated")
        assert not re.search(r"\bP\d{3}\b", response["answer"])


def test_v3_context_carryover_with_llm_and_reset(monkeypatch):
    client = app.test_client()
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    responses = {
        "Any nephew birthdays in September?": llm_result(
            "birthday_search", {"relationship": "Nephew", "month": 9}
        ),
        "What about October?": llm_result(
            "unknown", {"month": 10}
        ),
    }
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        lambda message: responses[message],
    )
    client.post(
        "/api/chat",
        json={"message": "Any nephew birthdays in September?"},
    )
    followup = client.post(
        "/api/chat", json={"message": "What about October?"}
    ).get_json()
    assert followup["debug"]["entities"]["month"] == 10
    assert followup["debug"]["entities"]["relationship"] == "Nephew"
    assert followup["debug"]["context_used"] is True
    client.post("/reset")
    reset_followup = client.post(
        "/api/chat", json={"message": "What about October?"}
    ).get_json()
    assert reset_followup["debug"]["intent"] == "unknown"
    assert not reset_followup["debug"].get("context_used")


def test_unknown_question_uses_structured_unknown_intent(monkeypatch):
    client = app.test_client()
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        lambda message: llm_result("unknown", {}),
    )
    response = client.post(
        "/api/chat", json={"message": "Tell me a poem about clouds"}
    ).get_json()
    assert response["debug"]["intent"] == "unknown"
    assert "rephrase" in response["answer"].lower()


@pytest.mark.parametrize(
    "message,intent,person_id",
    [
        ("Show birthday of P001.", "birthday_search", "P001"),
        ("Where does P001 stay?", "person_place", "P001"),
        ("Tell me about P001.", "person_info", "P001"),
        ("Who is P001?", "person_info", "P001"),
        (
            "Show the birthday of the person with ID P002.",
            "birthday_search",
            "P002",
        ),
    ],
)
def test_chat_person_id_requests_are_grounded_to_workbook(
    monkeypatch, message, intent, person_id
):
    monkeypatch.setenv("GROQ_API_KEY", "configured-for-mock")
    monkeypatch.setattr(
        app_module,
        "parse_with_llm",
        Mock(side_effect=AssertionError("Person_ID parsing must be deterministic")),
    )
    person = PEOPLE[PEOPLE["Person_ID"] == person_id].iloc[0]

    response = app.test_client().post(
        "/api/chat", json={"message": message}
    ).get_json()

    assert response["debug"]["intent"] == intent
    assert response["debug"]["entities"]["person_id"] == person_id
    assert response["debug"]["nlu_method"] == "Rule-based Person_ID lookup"
    assert person["Name"] in response["answer"]
    if intent == "birthday_search":
        assert person["DOB"].strftime("%d %B %Y") in response["answer"]
    elif intent == "person_place":
        assert person["Place_of_Stay"] in response["answer"]
    else:
        assert person["DOB"].strftime("%d %B %Y") in response["answer"]
        assert person["Place_of_Stay"] in response["answer"]


def test_bare_person_id_asks_for_requested_information(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    person = PEOPLE[PEOPLE["Person_ID"] == "P001"].iloc[0]

    response = app.test_client().post(
        "/api/chat", json={"message": "P001"}
    ).get_json()

    assert person["Name"] in response["answer"]
    assert "what would you like to know" in response["answer"].lower()
    assert response["debug"]["entities"]["person_id"] == "P001"


def test_invalid_person_id_does_not_run_a_broad_query(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = app.test_client().post(
        "/api/chat", json={"message": "Show birthday of P999."}
    ).get_json()

    assert "couldn't find p999" in response["answer"].lower()
    assert "no query executed" in response["debug"]["grounding_status"].lower()
    assert "person_id" not in response["debug"]["entities"]


def test_flask_secret_uses_environment_or_warns_and_generates_temporary_secret(
    monkeypatch,
):
    monkeypatch.setenv("FLASK_SECRET", "test-local-secret")
    assert app_module._flask_secret_from_environment() == "test-local-secret"

    monkeypatch.setenv("FLASK_SECRET", "replace_with_random_secret")
    with pytest.warns(RuntimeWarning, match="temporary development secret"):
        fallback_secret = app_module._flask_secret_from_environment()
    assert fallback_secret
    assert fallback_secret != "test-local-secret"

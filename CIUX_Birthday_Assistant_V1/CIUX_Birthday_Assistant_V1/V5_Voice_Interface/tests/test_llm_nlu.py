import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import llm_nlu
from llm_nlu import LLMNLUError, parse_with_llm


def fake_client(output_text):
    create = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=output_text)
                )
            ]
        )
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        ),
    )


def test_groq_json_is_parsed_and_model_is_configurable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    client = fake_client(
        json.dumps(
            {
                "intent": "birthday_search",
                "entities": {"relationship": "Nephew", "month": 9},
            }
        )
    )

    parsed = parse_with_llm("Do I need to wish any nephew during September?", client)

    assert parsed == {
        "intent": "birthday_search",
        "entities": {"relationship": "Nephew", "month": 9},
    }
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"][0]["role"] == "system"
    assert "Never answer" in kwargs["messages"][0]["content"]
    assert "test-key-not-a-real-secret" not in repr(kwargs)


def test_env_is_loaded_from_project_root_and_used_for_groq_client(
    monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=test-root-config-key\nGROQ_MODEL=root-config-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_nlu, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    client = fake_client('{"intent":"unknown","entities":{}}')
    groq_factory = Mock(return_value=client)
    monkeypatch.setattr(importlib.import_module("groq"), "Groq", groq_factory)

    parse_with_llm("anything")

    groq_factory.assert_called_once_with(
        api_key="test-root-config-key",
        timeout=15.0,
    )
    assert client.chat.completions.create.call_args.kwargs["model"] == (
        "root-config-model"
    )


def test_default_model_is_used_when_model_is_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr(llm_nlu, "load_dotenv", lambda **kwargs: None)
    client = fake_client('{"intent":"unknown","entities":{}}')

    parse_with_llm("anything", client)

    assert client.chat.completions.create.call_args.kwargs["model"] == (
        llm_nlu.DEFAULT_MODEL
    )


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"entities": {}}',
        '{"intent": "delete_everything", "entities": {}}',
        '{"intent": "unknown", "entities": {"person_id": "P004"}}',
        '{"intent": "unknown", "entities": {"month": 13}}',
        '{"intent": "unknown", "entities": []}',
    ],
)
def test_invalid_or_unexpected_responses_are_rejected(monkeypatch, response):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(LLMNLUError):
        parse_with_llm("Anything", fake_client(response))


def test_api_failure_is_reported_as_llm_nlu_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(side_effect=TimeoutError("timeout"))
            )
        )
    )
    with pytest.raises(LLMNLUError, match="Groq NLU timeout") as error:
        parse_with_llm("Upcoming birthdays", client)
    assert error.value.category == "timeout"
    assert "retry after checking connectivity" in error.value.diagnostic


def test_missing_groq_api_key_is_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_nlu, "load_dotenv", lambda **kwargs: None)
    with pytest.raises(LLMNLUError, match="GROQ_API_KEY is not configured"):
        parse_with_llm("Upcoming birthdays")


def test_example_key_placeholder_is_not_treated_as_a_configured_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    monkeypatch.setattr(llm_nlu, "load_dotenv", lambda **kwargs: None)

    with pytest.raises(LLMNLUError, match="GROQ_API_KEY is not configured"):
        parse_with_llm("Upcoming birthdays")

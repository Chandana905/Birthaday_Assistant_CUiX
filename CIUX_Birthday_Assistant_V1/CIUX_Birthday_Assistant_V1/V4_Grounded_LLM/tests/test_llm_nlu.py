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


def test_env_is_loaded_from_v4_root_and_configures_groq_client(
    monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=test-v4-root-key\nGROQ_MODEL=v4-test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_nlu, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    client = fake_client('{"intent":"unknown","entities":{}}')
    groq_factory = Mock(return_value=client)
    monkeypatch.setattr(
        importlib.import_module("groq"), "Groq", groq_factory
    )

    parse_with_llm("anything")

    groq_factory.assert_called_once_with(
        api_key="test-v4-root-key",
        timeout=15.0,
    )
    assert client.chat.completions.create.call_args.kwargs["model"] == (
        "v4-test-model"
    )
    assert "test-v4-root-key" not in repr(
        client.chat.completions.create.call_args
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
    with pytest.raises(LLMNLUError, match="request failed"):
        parse_with_llm("Upcoming birthdays", client)


def test_missing_groq_api_key_is_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_nlu, "load_dotenv", lambda **kwargs: None)
    with pytest.raises(LLMNLUError, match="GROQ_API_KEY is not configured"):
        parse_with_llm("Upcoming birthdays")

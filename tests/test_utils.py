"""Tests for src.utils."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.utils import ChatClient, Config, extract_final_message, load_config


class TestLoadConfig:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("PROJECT_ID", "123")
        config = load_config()
        assert config.host == "https://api.example.com/"
        assert config.api_key == "Bearer test-key"
        assert config.project_id == 123

    def test_adds_trailing_slash_to_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("API_KEY", "key")
        monkeypatch.setenv("PROJECT_ID", "1")
        config = load_config()
        assert config.host.endswith("/")

    def test_adds_bearer_to_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("API_KEY", "my-key")
        monkeypatch.setenv("PROJECT_ID", "1")
        config = load_config()
        assert config.api_key.startswith("Bearer ")

    def test_keeps_bearer_if_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("API_KEY", "Bearer existing")
        monkeypatch.setenv("PROJECT_ID", "1")
        config = load_config()
        assert config.api_key == "Bearer existing"

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_KEY", "")
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("PROJECT_ID", "1")
        with pytest.raises(RuntimeError, match="API_KEY"):
            load_config()

    def test_missing_project_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("API_KEY", "key")
        monkeypatch.setenv("PROJECT_ID", "")
        with pytest.raises(RuntimeError, match="PROJECT_ID"):
            load_config()

    def test_invalid_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "invalid")
        monkeypatch.setenv("API_KEY", "key")
        monkeypatch.setenv("PROJECT_ID", "1")
        with pytest.raises(RuntimeError, match="HOST"):
            load_config()

    def test_invalid_project_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "https://api.example.com")
        monkeypatch.setenv("API_KEY", "key")
        monkeypatch.setenv("PROJECT_ID", "not-a-number")
        with pytest.raises(RuntimeError, match="PROJECT_ID"):
            load_config()


class TestChatClient:
    def test_headers(self) -> None:
        config = Config(
            host="https://api.example.com/", api_key="Bearer x", project_id=1
        )
        client = ChatClient(config)
        headers = client._headers()
        assert headers["Authorization"] == "Bearer x"
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    def test_chat_payload(self) -> None:
        config = Config(
            host="https://api.example.com/", api_key="Bearer x", project_id=1
        )
        client = ChatClient(config)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        with patch("src.utils.requests.post", return_value=mock_response) as mock_post:
            client.chat("hello", stream=False)
            call_args = mock_post.call_args
            assert "api.example.com" in call_args[0][0]
            assert "/api/chat/" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["user_input"] == "hello"
            assert payload["projectId"] == 1

    def test_iter_ndjson_lines(self) -> None:
        config = Config(
            host="https://api.example.com/", api_key="Bearer x", project_id=1
        )
        client = ChatClient(config)
        mock_response = MagicMock()
        mock_response.iter_lines = lambda **kw: iter(
            [json.dumps({"type": "a", "x": 1}), "", json.dumps({"type": "b"})]
        )
        lines = list(client.iter_ndjson_lines(mock_response))
        assert len(lines) == 2
        assert lines[0] == {"type": "a", "x": 1}
        assert lines[1] == {"type": "b"}


class TestExtractFinalMessage:
    def test_finds_replace_message(self) -> None:
        events = [
            {"type": "other"},
            {"type": "replace_message", "message": "Final answer"},
        ]
        assert extract_final_message(events) == "Final answer"

    def test_last_replace_wins(self) -> None:
        events = [
            {"type": "replace_message", "message": "First"},
            {"type": "replace_message", "message": "Second"},
        ]
        assert extract_final_message(events) == "Second"

    def test_no_replace_message_returns_none(self) -> None:
        events = [{"type": "other"}, {"type": "another"}]
        assert extract_final_message(events) is None

    def test_empty_returns_none(self) -> None:
        assert extract_final_message([]) is None

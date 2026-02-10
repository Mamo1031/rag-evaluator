"""Tests for src.answer_generator."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.answer_generator import (
    Chunk,
    _build_index,
    _char_bigrams,
    _format_context,
    _generate_answer,
    _load_chunks,
    _load_questions,
    _retrieve_chunks,
    _split_text,
)


class TestCharBigrams:
    def test_basic(self) -> None:
        assert _char_bigrams("hello") == ["he", "el", "ll", "lo"]

    def test_removes_whitespace(self) -> None:
        assert _char_bigrams("a b") == ["ab"]

    def test_short_string(self) -> None:
        assert _char_bigrams("a") == []


class TestSplitText:
    def test_single_paragraph(self) -> None:
        text = "A" * 150
        chunks = _split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_paragraphs(self) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        chunks = _split_text(text, max_chars=250, min_chars=50)
        assert len(chunks) >= 1

    def test_min_chars_filter(self) -> None:
        text = "Short"
        chunks = _split_text(text, min_chars=100)
        assert len(chunks) == 0

    def test_splits_when_exceeds_max_chars(self) -> None:
        p1 = "A" * 100
        p2 = "B" * 100
        text = p1 + "\n\n" + p2
        chunks = _split_text(text, max_chars=150, min_chars=50)
        assert len(chunks) >= 2

    def test_skips_chunk_below_min_when_splitting(self) -> None:
        text = "A" * 50 + "\n\n" + "B" * 200
        chunks = _split_text(text, max_chars=100, min_chars=80)
        assert all(len(c) >= 80 for c in chunks)


class TestBuildIndex:
    def test_basic(self) -> None:
        raw = [
            {"source": "a.md", "text": "hello world"},
            {"source": "b.md", "text": "world"},
        ]
        chunks, idf = _build_index(raw)
        assert len(chunks) == 2
        assert len(idf) > 0
        assert all(c.norm > 0 for c in chunks)


class TestLoadChunks:
    def test_loads_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("Hello\n\nWorld " * 50, encoding="utf-8")
        chunks = _load_chunks(tmp_path)
        assert len(chunks) >= 1
        assert chunks[0]["source"] == "doc.md"

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        chunks = _load_chunks(tmp_path)
        assert chunks == []


class TestLoadQuestions:
    def test_loads_questions_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "questions.xlsx"
        df = pd.DataFrame({"question": ["Q1", "", "Q2"]})
        df.to_excel(path, index=False)
        questions = _load_questions(path)
        assert questions == ["Q1", "Q2"]

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.xlsx"
        with pytest.raises(RuntimeError, match="Questions file not found"):
            _load_questions(path)

    def test_raises_when_no_questions(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.xlsx"
        df = pd.DataFrame({"question": []})
        df.to_excel(path, index=False)
        with pytest.raises(RuntimeError, match="No questions found"):
            _load_questions(path)


class TestRetrieveChunks:
    def test_returns_top_k(self) -> None:
        chunk1 = Chunk("a", "text1", Counter(), 1, 1.0)
        chunk2 = Chunk("b", "text2", Counter(), 1, 1.0)
        chunk3 = Chunk("c", "text3", Counter(), 1, 1.0)
        idf: dict[str, float] = {}
        result = _retrieve_chunks("query", [chunk1, chunk2, chunk3], idf, top_k=2)
        assert len(result) <= 2

    def test_scores_by_similarity(self) -> None:
        raw = [
            {"source": "a.md", "text": "hello world"},
            {"source": "b.md", "text": "world peace"},
        ]
        chunks, idf = _build_index(raw)
        result = _retrieve_chunks("hello", chunks, idf, top_k=1)
        assert len(result) == 1
        assert result[0].source == "a.md"


class TestFormatContext:
    def test_formats_chunks(self) -> None:
        chunk = Chunk("doc.md", "content", Counter(), 1, 1.0)
        text, contexts = _format_context([chunk])
        assert "[1] doc.md" in text
        assert "content" in text
        assert len(contexts) == 1
        assert contexts[0]["source"] == "doc.md"
        assert contexts[0]["text"] == "content"


class TestGenerateAnswer:
    def test_returns_answer_from_mock_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()
        mock_client.iter_ndjson_lines.return_value = [
            {"type": "replace_message", "message": "回答です"}
        ]
        chunk = Chunk("doc.md", "content", Counter(), 1, 1.0)
        answer = _generate_answer(
            mock_client, "質問", [chunk], timeout=10, max_retries=1
        )
        assert answer == "回答です"

    def test_raises_after_retries_on_request_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests

        import src.answer_generator as ag

        mock_client = MagicMock()
        mock_client.chat.side_effect = requests.RequestException("boom")
        mock_client.iter_ndjson_lines.return_value = []
        monkeypatch.setattr(ag, "time", MagicMock(sleep=lambda *_args, **_kwargs: None))

        with pytest.raises(RuntimeError, match="API call failed"):
            _generate_answer(
                mock_client,
                "質問",
                [Chunk("doc.md", "content", Counter(), 1, 1.0)],
                timeout=10,
                max_retries=2,
            )


class TestMain:
    def test_main_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import src.answer_generator as ag

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        monkeypatch.setenv("DOCS_DIR", str(docs_dir))

        # Avoid real IO and heavy work in tests
        monkeypatch.setattr(
            ag,
            "_load_chunks",
            lambda _path: [{"source": "doc.md", "text": "hello world"}],
        )

        sample_chunk = Chunk("doc.md", "content", Counter(), 1, 1.0)
        monkeypatch.setattr(
            ag,
            "_build_index",
            lambda _raw: ([sample_chunk], {}),
        )

        monkeypatch.setattr(ag, "load_config", lambda: {"dummy": True})

        class DummyClient:
            def __init__(self, _config: dict) -> None:
                self.config = _config

            def chat(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused
                return None

            def iter_ndjson_lines(self, _response) -> list[dict]:  # pragma: no cover
                return []

        monkeypatch.setattr(ag, "ChatClient", DummyClient)
        monkeypatch.setattr(
            ag,
            "_retrieve_chunks",
            lambda question, chunks, idf, top_k=5: [sample_chunk],
        )
        monkeypatch.setattr(
            ag,
            "_generate_answer",
            lambda client, question, chunks, timeout, max_retries: "answer",
        )
        monkeypatch.setattr(ag.time, "sleep", lambda *_args, **_kwargs: None)

        questions_path = tmp_path / "questions.xlsx"
        pd.DataFrame({"question": ["Q1"]}).to_excel(questions_path, index=False)
        monkeypatch.setenv("QUESTIONS_PATH", str(questions_path))

        output_path = tmp_path / "out.xlsx"
        monkeypatch.setenv("OUTPUT_PATH", str(output_path))

        exit_code = ag.main()
        assert exit_code == 0
        assert output_path.exists()

    def test_main_raises_when_docs_dir_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.answer_generator as ag

        monkeypatch.delenv("DOCS_DIR", raising=False)
        with pytest.raises(RuntimeError, match="DOCS_DIR is not set"):
            ag.main()

    def test_main_raises_when_docs_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import src.answer_generator as ag

        missing_dir = tmp_path / "missing"
        monkeypatch.setenv("DOCS_DIR", str(missing_dir))
        with pytest.raises(RuntimeError, match="docs directory not found"):
            ag.main()

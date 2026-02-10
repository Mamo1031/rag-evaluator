"""Tests for src.qa_pair_generator."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestParseQuestions:
    def test_normalizes_numbering_and_blanks(self) -> None:
        from src.qa_pair_generator import _parse_questions

        raw = (
            "\n1. 休暇申請の手順は？\n"
            "- 在宅勤務の申請方法は？\n"
            "\nQ3: 遅刻時の連絡先は？\n"
        )
        questions = _parse_questions(raw)
        assert questions == [
            "休暇申請の手順は？",
            "在宅勤務の申請方法は？",
            "遅刻時の連絡先は？",
        ]


class TestGenerateQuestions:
    def test_supplements_until_target_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.qa_pair_generator as qg

        calls = {"count": 0}
        responses = iter(
            [
                "1. 休暇申請の手順は？\n2. 在宅勤務の申請方法は？",
                "1. 休暇申請の手順は？\n2. 遅刻時の連絡先は？",
            ]
        )

        def fake_chat_text(*_args, **_kwargs) -> str:
            calls["count"] += 1
            return next(responses)

        monkeypatch.setattr(qg, "_chat_text", fake_chat_text)

        questions = qg._generate_questions(
            client=object(),  # type: ignore[arg-type]
            raw_chunks=[{"source": "manual.md", "text": "A" * 500}],
            target_count=3,
            timeout=10,
            max_retries=1,
            rounds=3,
        )
        assert len(questions) == 3
        assert len(set(questions)) == 3
        assert calls["count"] == 2


class TestMain:
    def test_writes_aligned_question_answer_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import src.qa_pair_generator as qg

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        monkeypatch.setenv("DOCS_DIR", str(docs_dir))

        monkeypatch.setattr(
            qg,
            "_load_chunks",
            lambda _path: [{"source": "doc.md", "text": "manual content"}],
        )
        monkeypatch.setattr(qg, "_build_index", lambda _raw: (["chunk"], {}))
        monkeypatch.setattr(
            qg,
            "_generate_questions",
            lambda *args, **kwargs: ["Q1", "Q2", "Q3"],
        )
        monkeypatch.setattr(
            qg,
            "_retrieve_chunks",
            lambda question, chunks, idf, top_k=5: ["selected"],
        )
        monkeypatch.setattr(
            qg,
            "_generate_answer",
            lambda client, question, chunks, timeout, max_retries: "A",
        )
        monkeypatch.setattr(qg, "load_config", lambda: {"dummy": True})
        monkeypatch.setattr(qg.time, "sleep", lambda *_args, **_kwargs: None)

        class DummyClient:
            def __init__(self, _config: dict) -> None:
                self.config = _config

        monkeypatch.setattr(qg, "ChatClient", DummyClient)

        q_path = tmp_path / "q.txt"
        a_path = tmp_path / "a.txt"
        exit_code = qg.main(
            [
                "--count",
                "3",
                "--questions-out",
                str(q_path),
                "--answers-out",
                str(a_path),
            ]
        )

        assert exit_code == 0
        assert q_path.exists()
        assert a_path.exists()
        assert q_path.read_text(encoding="utf-8").splitlines() == ["Q1", "Q2", "Q3"]
        assert a_path.read_text(encoding="utf-8").splitlines() == ["A", "A", "A"]

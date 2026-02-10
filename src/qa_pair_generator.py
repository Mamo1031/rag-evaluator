"""Generate QA pairs from documents for RAG evaluation."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.answer_generator import (
    _build_index,
    _generate_answer,
    _load_chunks,
    _retrieve_chunks,
)
from src.utils import ChatClient, extract_final_message, load_config

DEFAULT_COUNT = 100
DEFAULT_QUESTIONS_OUT = Path("data/qa/question/q_auto100_office_manual.txt")
DEFAULT_ANSWERS_OUT = Path("data/qa/answer/a_auto100_office_manual.txt")


def _build_question_context(raw_chunks: list[dict], max_chars: int = 12000) -> str:
    blocks: list[str] = []
    total = 0
    for idx, chunk in enumerate(raw_chunks, start=1):
        block = f"[{idx}] {chunk['source']}\n{chunk['text']}"
        if total + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        total += len(block) + 2
    return "\n\n".join(blocks)


def _normalize_question_line(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    text = re.sub(
        r"^\s*(?:Q\s*\d+\s*[:：.]|\d+\s*[.)\]:：、]|[-*・]+)\s*",
        "",
        text,
    )
    text = text.strip().strip('"').strip("'").strip()
    return text


def _question_key(question: str) -> str:
    return re.sub(r"\s+", "", question).lower()


def _parse_questions(raw_text: str) -> list[str]:
    questions: list[str] = []
    for line in raw_text.splitlines():
        question = _normalize_question_line(line)
        if question:
            questions.append(question)
    return questions


def _chat_text(
    client: ChatClient,
    *,
    user_input: str,
    template: str,
    timeout: float,
    max_retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                user_input,
                template=template,
                stream=True,
                timeout=timeout,
            )
            events = list(client.iter_ndjson_lines(response))
            message = extract_final_message(events)
            return (message or "").strip()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 5))
    if last_error:
        raise RuntimeError(
            "API call failed while generating questions."
        ) from last_error
    return ""


def _generate_questions(
    client: ChatClient,
    raw_chunks: list[dict],
    *,
    target_count: int,
    timeout: float,
    max_retries: int,
    rounds: int = 6,
) -> list[str]:
    context_text = _build_question_context(raw_chunks)
    if not context_text:
        raise RuntimeError("No context for question generation.")

    template = (
        "あなたはドキュメント根拠の評価用質問を作るアシスタントです。"
        "入力された資料抜粋の内容だけで答えられる、日本語の質問を作成してください。"
        "質問は簡潔で具体的にし、1行に1問だけ出力してください。"
        "番号、箇条書き記号、前置き、解説、カテゴリ見出しは出力しないでください。"
    )

    questions: list[str] = []
    seen: set[str] = set()
    for round_idx in range(1, rounds + 1):
        remaining = target_count - len(questions)
        if remaining <= 0:
            break
        existing = "\n".join(f"- {q}" for q in questions) if questions else "(なし)"
        user_input = (
            f"不足件数: {remaining}\n"
            f"以下の資料抜粋を根拠に、重複しない質問を{remaining}件生成してください。\n"
            "既存質問と同じ内容や言い換えに近いものは避けてください。\n\n"
            f"既存質問:\n{existing}\n\n"
            f"資料抜粋:\n{context_text}"
        )
        raw = _chat_text(
            client,
            user_input=user_input,
            template=template,
            timeout=timeout,
            max_retries=max_retries,
        )
        parsed = _parse_questions(raw)
        for question in parsed:
            key = _question_key(question)
            if key in seen:
                continue
            seen.add(key)
            questions.append(question)
            if len(questions) >= target_count:
                break
        if round_idx < rounds and len(questions) < target_count:
            time.sleep(0.8)

    if len(questions) < target_count:
        message = (
            "Failed to generate enough unique questions: "
            f"{len(questions)}/{target_count}"
        )
        raise RuntimeError(message)
    return questions[:target_count]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate QA pairs from documents (questions + answers as text files)."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of questions/answers to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--questions-out",
        type=Path,
        default=DEFAULT_QUESTIONS_OUT,
        help=f"Output path for generated questions (default: {DEFAULT_QUESTIONS_OUT})",
    )
    parser.add_argument(
        "--answers-out",
        type=Path,
        default=DEFAULT_ANSWERS_OUT,
        help=f"Output path for generated answers (default: {DEFAULT_ANSWERS_OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.count <= 0:
        raise RuntimeError("--count must be a positive integer.")

    load_dotenv(override=True)
    docs_dir_env = os.getenv("DOCS_DIR")
    if not docs_dir_env:
        raise RuntimeError("DOCS_DIR is not set. Please configure it in .env.")
    docs_dir = Path(docs_dir_env).expanduser()
    if not docs_dir.exists():
        raise RuntimeError(f"docs directory not found: {docs_dir}")

    raw_chunks = _load_chunks(docs_dir)
    if not raw_chunks:
        raise RuntimeError(f"failed to extract chunks from docs: {docs_dir}")
    chunks, idf = _build_index(raw_chunks)

    config = load_config()
    client = ChatClient(config)
    timeout = float(os.getenv("REQUEST_TIMEOUT", "60"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))

    print(f"Generating {args.count} questions...")
    questions = _generate_questions(
        client,
        raw_chunks,
        target_count=args.count,
        timeout=timeout,
        max_retries=max_retries,
    )

    answers: list[str] = []
    for idx, question in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] Generating answer...")
        selected = _retrieve_chunks(question, chunks, idf, top_k=5)
        answer = _generate_answer(
            client,
            question,
            selected,
            timeout=timeout,
            max_retries=max_retries,
        )
        answers.append(answer)
        if idx < len(questions):
            time.sleep(1.1)

    args.questions_out.parent.mkdir(parents=True, exist_ok=True)
    args.answers_out.parent.mkdir(parents=True, exist_ok=True)
    args.questions_out.write_text("\n".join(questions), encoding="utf-8")
    args.answers_out.write_text("\n".join(answers), encoding="utf-8")
    print(f"Saved questions to: {args.questions_out}")
    print(f"Saved answers to: {args.answers_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

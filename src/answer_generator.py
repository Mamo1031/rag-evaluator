"""Generate reference answers from documents and question list."""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.utils import ChatClient, extract_final_message, load_config


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    counts: Counter
    total: int
    norm: float


def _char_bigrams(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", "", text)
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def _split_text(text: str, max_chars: int = 900, min_chars: int = 120) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if size + len(paragraph) + 1 > max_chars and buffer:
            combined = "\n".join(buffer).strip()
            if len(combined) >= min_chars:
                chunks.append(combined)
            buffer = []
            size = 0
        buffer.append(paragraph)
        size += len(paragraph) + 1
    if buffer:
        combined = "\n".join(buffer).strip()
        if len(combined) >= min_chars:
            chunks.append(combined)
    return chunks


def _load_chunks(docs_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for path in docs_dir.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in _split_text(text):
            chunks.append({"source": path.name, "text": chunk})
    return chunks


def _load_questions(path: Path) -> list[str]:
    """Load questions from a text or Excel file.

    - If the path ends with ``.xlsx``, the function expects an Excel file with a
      ``question`` column and loads non-empty values as questions.
    - Otherwise, it treats the file as UTF-8 text, one question per line.
    """
    if not path.exists():
        raise RuntimeError(f"Questions file not found: {path}")

    questions: list[str]
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
        if "question" not in df.columns:
            raise RuntimeError(f"No questions found in questions file: {path}")
        raw_values = df["question"].tolist()
        questions = []
        for value in raw_values:
            # Skip missing values (NaN, None, etc.)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                questions.append(text)
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        questions = [line.strip() for line in lines if line.strip()]

    if not questions:
        raise RuntimeError(f"No questions found in questions file: {path}")
    return questions


def _build_index(raw_chunks: Iterable[dict]) -> tuple[list[Chunk], dict[str, float]]:
    doc_freq: Counter = Counter()
    chunk_records: list[dict] = []
    for chunk in raw_chunks:
        bigrams = _char_bigrams(chunk["text"])
        counts = Counter(bigrams)
        total = max(sum(counts.values()), 1)
        for bg in counts.keys():
            doc_freq[bg] += 1
        chunk_records.append(
            {
                "source": chunk["source"],
                "text": chunk["text"],
                "counts": counts,
                "total": total,
            }
        )
    total_docs = max(len(chunk_records), 1)
    idf = {bg: math.log((total_docs + 1) / (df + 1)) + 1 for bg, df in doc_freq.items()}
    indexed: list[Chunk] = []
    for record in chunk_records:
        counts = record["counts"]
        total = record["total"]
        norm = math.sqrt(
            sum(((counts[bg] / total) * idf[bg]) ** 2 for bg in counts.keys())
        )
        indexed.append(
            Chunk(
                source=record["source"],
                text=record["text"],
                counts=counts,
                total=total,
                norm=norm or 1.0,
            )
        )
    return indexed, idf


def _retrieve_chunks(
    question: str, chunks: list[Chunk], idf: dict[str, float], top_k: int = 5
) -> list[Chunk]:
    query_bigrams = _char_bigrams(question)
    query_counts = Counter(query_bigrams)
    query_total = max(sum(query_counts.values()), 1)
    query_weights = {
        bg: (query_counts[bg] / query_total) * idf.get(bg, 0.0) for bg in query_counts
    }
    query_norm = math.sqrt(sum(w**2 for w in query_weights.values())) or 1.0

    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        dot = 0.0
        for bg, w_q in query_weights.items():
            if bg not in chunk.counts:
                continue
            w_d = (chunk.counts[bg] / chunk.total) * idf.get(bg, 0.0)
            dot += w_q * w_d
        score = dot / (query_norm * chunk.norm)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


def _format_context(chunks: list[Chunk]) -> tuple[str, list[dict]]:
    context_blocks = []
    contexts = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] {chunk.source}\n{chunk.text}")
        contexts.append({"source": chunk.source, "text": chunk.text})
    return "\n\n".join(context_blocks), contexts


def _generate_answer(
    client: ChatClient,
    question: str,
    chunks: list[Chunk],
    *,
    timeout: float,
    max_retries: int,
) -> str:
    context_text, _ = _format_context(chunks)
    prompt = (
        "以下の資料抜粋のみを根拠に、日本語で簡潔に回答してください。"
        "資料に記載がない場合は「資料に記載がありません」とだけ答えてください。"
        "回答は最大3文程度とし、不要な前置きや注意書き、箇条書きは書かないでください。"
    )
    user_input = f"{question}\n\n資料抜粋:\n{context_text}"
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                user_input,
                template=prompt,
                stream=True,
                timeout=timeout,
            )
            events = list(client.iter_ndjson_lines(response))
            answer = extract_final_message(events)
            answer_text = (answer or "").strip()
            if not answer_text:
                return ""
            sentences = re.split(r"(?<=。)", answer_text)
            concise = "".join(sentences[:3]).strip()
            return concise
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 5))
    if last_error:
        raise RuntimeError("API call failed.") from last_error
    return ""


def main(questions_path: Path | None = None, output_path: Path | None = None) -> int:
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

    # Resolve questions / output paths
    if questions_path is None:
        questions_env = os.getenv("QUESTIONS_PATH")
        if not questions_env:
            raise RuntimeError("QUESTIONS_PATH is not set.")
        questions_path = Path(questions_env)
    if output_path is None:
        output_env = os.getenv("OUTPUT_PATH")
        if not output_env:
            raise RuntimeError("OUTPUT_PATH is not set.")
        output_path = Path(output_env)

    # Load questions
    questions = _load_questions(questions_path)

    timeout = float(os.getenv("REQUEST_TIMEOUT", "60"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
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

    # Write answers as plain text, one per line
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(answers), encoding="utf-8")
    print(f"Saved answers to: {output_path}")
    return 0


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 3:
        message = (
            "Usage:\n"
            "  python -m src.answer_generator "
            "<questions_txt_path> <answers_txt_path>\n\n"
            "Arguments:\n"
            "  <questions_txt_path>  Path to a UTF-8 text file.\n"
            "  <answers_txt_path>    Path to the output text file.\n"
        )
        raise SystemExit(message)
    questions_arg = Path(argv[1])
    answers_arg = Path(argv[2])
    raise SystemExit(main(questions_arg, answers_arg))

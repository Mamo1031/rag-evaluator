# rag-evaluator

[![CI](https://github.com/Mamo1031/rag-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/Mamo1031/rag-evaluator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Mamo1031/rag-evaluator/graph/badge.svg)](https://codecov.io/gh/Mamo1031/rag-evaluator)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Utilities for generating reference answers from documents via an API and using them for RAG (Retrieval-Augmented Generation) evaluation.

## Features

- **Answer generation** (`src/answer_generator.py`): Chunk documents, retrieve relevant chunks, and generate answers via API
- **QA pair generation** (`src/qa_pair_generator.py`): Skeleton implementation (to be extended)

## Project Structure

```
rag-evaluator/
├── data/
│   ├── reference/                  # Source documents
│   └── qa/
│       ├── question/               # Question text files (one per line)
│       └── answer/                 # Generated answer text files
├── scripts/
│   └── check.py                    # CLI for lint/format/test
├── src/
│   ├── answer_generator.py         # Answer generation from docs + questions
│   ├── qa_pair_generator.py        # Skeleton (for future QA pair generation)
│   ├── evaluator.py                # Skeleton (for future evaluation logic)
│   └── utils.py                    # API client / config helpers
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/install/)

## Setup

```bash
uv venv
uv sync --all-extras
source .venv/bin/activate
```

## Environment Variables

Create a `.env` file in the project root.

| Variable            | Required | Description                                            |
| ------------------- | -------- | ------------------------------------------------------ |
| `HOST`            | Yes      | API base URL (e.g.`https://api.example.com`)         |
| `API_KEY`         | Yes      | API key (Bearer prefix is added automatically)         |
| `PROJECT_ID`      | Yes      | Project ID (integer)                                   |
| `DOCS_DIR`        | Yes      | Directory containing source documents (Markdown files) |
| `REQUEST_TIMEOUT` | No       | API timeout in seconds (default:`60`)                |
| `MAX_RETRIES`     | No       | Number of retries on API failure (default:`3`)       |

Example:

```
HOST=https://api.example.com
API_KEY=your_api_key_here
PROJECT_ID=1
```

## Usage

### QA pair generation (100 questions + 100 answers)

Generate 100 document-grounded questions and their answers in one run:

```bash
uv run python -m src.qa_pair_generator \
  --count 100 \
  --questions-out data/qa/question/q_auto100_office_manual.txt \
  --answers-out data/qa/answer/a_auto100_office_manual.txt
```

Output:

- `--questions-out`: UTF-8 text file, one question per line
- `--answers-out`: UTF-8 text file, one answer per line
- Line numbers are aligned between files (same line index = one QA pair)

### Answer generation

1. Place Markdown documents in the directory specified by `DOCS_DIR` (e.g. `data/reference/docs`).
2. Prepare a questions text file (UTF-8), one question per line, for example:

   - `data/qa/question/questions.txt`
3. Run the answer generator from the project root:

```bash
uv run python -m src.answer_generator \
  data/qa/question/questions.txt \
  data/qa/answer/answers.txt
```

Output:

- A UTF-8 text file is written to the path you pass as `<answers_txt_path>` (one answer per line, aligned with the input questions).

## Development

### Code quality checks

This project uses the following tools:

- Ruff: Linter / formatter
- mypy: Static type checking
- pydoclint: Docstring style checking
- pytest + pytest-cov: Testing and coverage

Run code quality checks (with auto-fix):

```bash
uv run poe lint
```

Run tests:

```bash
uv run poe test
```

Run all checks:

```bash
uv run poe all
```
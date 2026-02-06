"""CLI for lint, format, and test."""

from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> int:
    result = subprocess.run(cmd)
    return result.returncode


def _lint(*, fix: bool = False) -> int:
    cmd = [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"]
    if fix:
        cmd.append("--fix")
    return _run(cmd)


def _format(*, check_only: bool = False) -> int:
    cmd = [sys.executable, "-m", "ruff", "format", "src", "scripts", "tests"]
    if check_only:
        cmd.append("--check")
    return _run(cmd)


def _test(*, coverage: bool = False) -> int:
    cmd = [sys.executable, "-m", "pytest", "tests", "-v"]
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    return _run(cmd)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check <lint|format|test|all> [options]")
        print("  lint   - Run ruff check (use --fix to auto-fix)")
        print("  format - Run ruff format (use --check to check only)")
        print("  test   - Run pytest (use --cov for coverage)")
        print("  all    - Run format, lint, test (with coverage)")
        return 1

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "lint":
        return _lint(fix="--fix" in args)
    if command == "format":
        return _format(check_only="--check" in args)
    if command == "test":
        return _test(coverage="--cov" in args or "--coverage" in args)
    if command == "all":
        if _format() != 0:
            return 1
        if _lint(fix=True) != 0:
            return 1
        return _test(coverage=True)

    print(f"Unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

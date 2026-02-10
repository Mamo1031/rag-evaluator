from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def _run(cmd: List[str]) -> int:
    """Run a command and return its exit code."""
    print(f"+ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        return 1
    return result.returncode


def cmd_lint() -> int:
    """Run linting / formatting / static analysis."""
    exit_code = 0

    # Ruff: lint (with auto-fix) + format
    if (code := _run(["ruff", "check", "--fix", "src", "scripts", "tests"])) != 0:
        exit_code = code
    if (code := _run(["ruff", "format", "src", "scripts", "tests"])) != 0:
        exit_code = code

    # mypy: type checking
    if (code := _run(["mypy", "src", "tests"])) != 0:
        exit_code = code

    # pydoclint: docstring check
    if (code := _run(["pydoclint", "src", "tests"])) != 0:
        exit_code = code

    return exit_code


def cmd_test() -> int:
    """Run tests with coverage."""
    return _run(["pytest", "tests", "-v", "--cov=src"])


def cmd_all() -> int:
    """Run all checks (lint + test)."""
    code_lint = cmd_lint()
    code_test = cmd_test()
    return 0 if code_lint == 0 and code_test == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Code quality and test helper CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("lint", help="Run lint / format / static analysis.")
    subparsers.add_parser("test", help="Run tests with coverage.")
    subparsers.add_parser("all", help="Run all checks (lint + test).")

    args = parser.parse_args(argv)

    if args.command == "lint":
        return cmd_lint()
    if args.command == "test":
        return cmd_test()
    if args.command == "all":
        return cmd_all()

    # Fallback (should not be reached)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

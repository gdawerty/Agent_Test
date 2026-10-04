#!/usr/bin/env python3
"""Small GitHub issue-to-PR coding agent.

The model can inspect and edit the checkout through a deliberately small tool
surface. Git and GitHub side effects stay in this process so the model cannot
commit, push, or open a pull request by itself.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, Sequence


DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-5.6"
DEFAULT_ENGINE = "custom"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_TOOL_OUTPUT = 8_000
MAX_FILE_OUTPUT = 8_000
MAX_STEPS = 12
MAX_READ_LINES = 200
GEMINI_REQUEST_DELAY_SECONDS = 5.0
PHASE_BUDGETS = {
    "investigate": 4,
    "implement": 3,
    "repair": 3,
    "verify": 2,
}
REPOSITORY_INDEX_VERSION = 1
MAX_INDEX_SYMBOLS = 250
MINI_DEFAULT_IMAGE = "agent-fix-sandbox:latest"
MINI_DEFAULT_COST_LIMIT = 3.0
MINI_MAX_CALLS = 12
MINI_FALLBACK_ENABLED = False
MINI_EDIT_DEADLINE_CALL = 4
MINI_STALL_CALLS = 3
TEST_ENV_ALLOWLIST = {
    "CI",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
}
PROTECTED_PATH_PREFIXES = (
    ".github/workflows/",
    ".github/actions/",
)
PROTECTED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    "credentials",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
}
PROTECTED_FILE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}
MINI_IGNORED_DIRECTORIES = IGNORED_DIRECTORIES | {
    ".aws",
    ".config",
    ".gnupg",
    ".secrets",
    ".ssh",
    "secrets",
}


class AgentError(RuntimeError):
    """An expected failure that should be shown as a user-facing CLI error."""


class MiniEditDeadlineExceeded(RuntimeError):
    """The mini engine stalled after its investigation checkpoint without editing."""


@dataclass
class RunBudget:
    """One model-call budget shared by every phase and engine in a run."""

    max_model_calls: int
    model_calls_used: int = 0
    calls_by_phase: dict[str, int] = field(default_factory=dict)
    phase_limits: dict[str, int] = field(
        default_factory=lambda: dict(PHASE_BUDGETS)
    )

    @property
    def remaining(self) -> int:
        return max(0, self.max_model_calls - self.model_calls_used)

    def phase_remaining(self, phase: str) -> int:
        limit = self.phase_limits.get(phase)
        if limit is None:
            return self.remaining
        return max(0, limit - self.calls_by_phase.get(phase, 0))

    def consume(self, count: int = 1, *, phase: str) -> None:
        if count < 0:
            raise ValueError("count must not be negative")
        phase_limit = self.phase_limits.get(phase)
        phase_used = self.calls_by_phase.get(phase, 0)
        if phase_limit is not None and phase_used + count > phase_limit:
            raise AgentError(
                f"{phase} phase budget exhausted "
                f"({phase_used}/{phase_limit} calls used)."
            )
        if self.model_calls_used + count > self.max_model_calls:
            raise AgentError(
                f"Global model-call budget exhausted before {phase} could continue "
                f"({self.model_calls_used}/{self.max_model_calls} calls used)."
            )
        self.model_calls_used += count
        self.calls_by_phase[phase] = self.calls_by_phase.get(phase, 0) + count


@dataclass(frozen=True)
class RepositorySymbol:
    path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str


@dataclass(frozen=True)
class RepositoryIndex:
    """Compact, deterministic repository structure used before model exploration."""

    revision: str
    files: tuple[str, ...]
    symbols: tuple[RepositorySymbol, ...]

    def render(self, issue_text: str = "", limit: int = MAX_TOOL_OUTPUT) -> str:
        terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue_text)
        }

        def score(symbol: RepositorySymbol) -> tuple[int, str, int]:
            path = symbol.path.lower()
            name = symbol.name.lower()
            score_value = 0
            for term in terms:
                if term == name:
                    score_value += 8
                elif term in name:
                    score_value += 4
                elif term in path:
                    score_value += 2
            if "test" in path or symbol.kind == "test":
                score_value += 1
            return (-score_value, symbol.path, symbol.start_line)

        ranked_symbols = sorted(self.symbols, key=score)[:MAX_INDEX_SYMBOLS]
        lines = [
            f"Repository map for revision {self.revision}.",
            "The map is generated before the agent starts; use it to target reads.",
            "",
            "Files:",
            *(f"- {path}" for path in self.files[:1000]),
            "",
            "Python symbols (path:line-end kind name signature):",
        ]
        lines.extend(
            f"- {symbol.path}:{symbol.start_line}-{symbol.end_line} "
            f"{symbol.kind} {symbol.name} {symbol.signature}"
            for symbol in ranked_symbols
        )
        if not self.symbols:
            lines.append("(no Python symbols indexed)")
        return _truncate("\n".join(lines), limit)


_REPOSITORY_INDEX_CACHE: dict[tuple[str, str, int], RepositoryIndex] = {}


def _repository_revision(root: Path) -> str:
    revision = _run_process(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False
    ).strip()
    return revision if re.fullmatch(r"[0-9a-f]{7,40}", revision) else "working-tree"


def _index_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if _is_secret_path(relative):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _python_symbols(root: Path, path: Path) -> list[RepositorySymbol]:
    relative = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    symbols: list[RepositorySymbol] = []

    def visit(nodes: list[ast.AST], parent: str | None = None) -> None:
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets: list[ast.AST] = []
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                else:
                    targets.append(node.target)
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if parent is None and not target.id.isupper():
                        continue
                    name = f"{parent}.{target.id}" if parent else target.id
                    symbols.append(
                        RepositorySymbol(
                            relative,
                            name,
                            "constant" if parent is None else "attribute",
                            node.lineno,
                            getattr(node, "end_lineno", node.lineno),
                            name,
                        )
                    )
            elif isinstance(node, ast.ClassDef):
                name = f"{parent}.{node.name}" if parent else node.name
                symbols.append(
                    RepositorySymbol(
                        relative,
                        name,
                        "class",
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        f"class {node.name}",
                    )
                )
                visit(node.body, name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{parent}.{node.name}" if parent else node.name
                try:
                    arguments = ast.unparse(node.args)
                except Exception:
                    arguments = "(...)"
                kind = "method" if parent else "function"
                if name.startswith("test") or ".test_" in name:
                    kind = "test"
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                symbols.append(
                    RepositorySymbol(
                        relative,
                        name,
                        kind,
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        f"{prefix} {node.name}({arguments})",
                    )
                )
                # Nested functions are implementation details; their parent is
                # enough for the first-pass map and keeps the context compact.

    visit(tree.body)
    return symbols


def build_repository_index(root: Path) -> RepositoryIndex:
    """Build or reuse a structural map for the checked-out revision."""
    root = root.resolve()
    revision = _repository_revision(root)
    key = (str(root), revision, REPOSITORY_INDEX_VERSION)
    cached = _REPOSITORY_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    paths = _index_files(root)
    files = tuple(path.relative_to(root).as_posix() for path in paths)
    symbols: list[RepositorySymbol] = []
    for path in paths:
        if path.suffix.lower() in {".py", ".pyi"}:
            symbols.extend(_python_symbols(root, path))
    index = RepositoryIndex(revision, files, tuple(symbols))
    _REPOSITORY_INDEX_CACHE[key] = index
    return index


def _truncate(value: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated at {limit} characters]"


def _rate_limit_delay(error: Exception) -> float | None:
    """Return a retry delay for a recoverable per-minute rate-limit error."""
    message = str(error)
    if "429" not in message and "rate" not in message.lower():
        return None

    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    if "perminute" in message.lower() or "per minute" in message.lower():
        return 60.0
    return None


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    check: bool = True,
) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise AgentError(f"Command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        raise AgentError(
            f"Command timed out after {timeout} seconds: {' '.join(command)}\n{output}"
        ) from exc

    output = (result.stdout or "") + (result.stderr or "")
    if check and result.returncode != 0:
        raise AgentError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"{_truncate(output)}"
        )
    return output


def _test_environment() -> dict[str, str]:
    """Run repository tests with a minimal environment and no credentials."""
    return {
        name: os.environ[name]
        for name in TEST_ENV_ALLOWLIST
        if name in os.environ
    }


def _is_protected_path(relative: Path) -> bool:
    """Return whether an agent edit could affect workflow execution or secrets."""
    normalized = relative.as_posix()
    filename = relative.name.lower()
    return (
        any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix)
            for prefix in PROTECTED_PATH_PREFIXES
        )
        or filename in PROTECTED_FILE_NAMES
        or filename.startswith(".env.")
        or filename.endswith(PROTECTED_FILE_SUFFIXES)
    )


def _is_secret_path(relative: Path) -> bool:
    """Return whether a path looks like a credential or private key file."""
    filename = relative.name.lower()
    return (
        filename in PROTECTED_FILE_NAMES
        or filename.startswith(".env")
        or filename.endswith(PROTECTED_FILE_SUFFIXES)
    )


class RepoWorkspace:
    """The model-facing operations for one checked-out repository."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _safe_path(self, relative_path: str) -> Path:
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("Path must be a non-empty repository-relative path")

        target = (self.root / relative_path).resolve()
        try:
            relative = target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path is outside the repository") from exc

        if ".git" in relative.parts:
            raise ValueError("Access to .git is not allowed")
        return target

    def list_files(self) -> str:
        files: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            if any(part in IGNORED_DIRECTORIES for part in relative.parts):
                continue
            if _is_secret_path(relative):
                continue
            files.append(relative.as_posix())

        files.sort()
        result = "\n".join(files[:1000])
        if len(files) > 1000:
            result += f"\n...[{len(files) - 1000} more files omitted]"
        return result or "(repository has no visible files)"

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        target = self._safe_path(path)
        relative = target.relative_to(self.root)
        if _is_secret_path(relative):
            return f"ERROR: access to credential file {path} is blocked"
        if not target.exists():
            return f"ERROR: {path} does not exist"
        if not target.is_file():
            return f"ERROR: {path} is not a file"
        if start_line < 1:
            return "ERROR: start_line must be at least 1"
        if end_line is not None and end_line < start_line:
            return "ERROR: end_line must be greater than or equal to start_line"

        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return f"ERROR: {path} is not a UTF-8 text file"

        start = start_line - 1
        requested_end = end_line if end_line is not None else start_line + MAX_READ_LINES - 1
        end = min(requested_end, start_line + MAX_READ_LINES - 1, len(lines))
        if start >= len(lines):
            return f"ERROR: start_line {start_line} is past the end of {path}"

        selected = [
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(start_line, end + 1)
        ]
        result = "\n".join(selected)
        if end < requested_end and end < len(lines):
            result += f"\n...[read limited to {MAX_READ_LINES} lines]"
        return _truncate(result, MAX_FILE_OUTPUT)

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        target = self._safe_path(path)
        relative = target.relative_to(self.root)
        if _is_protected_path(relative):
            raise ValueError(
                "Editing workflow, action, credential, or private-key files is blocked"
            )
        if len(new_text) > 1_000_000:
            raise ValueError("Refusing to write more than 1 MB")

        if not target.exists():
            if old_text:
                return "ERROR: file does not exist; use an empty old_text to create it"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding="utf-8")
            return f"Successfully created {path} ({len(new_text)} characters)"

        if not target.is_file():
            return f"ERROR: {path} is not a file"
        if not old_text:
            return "ERROR: old_text must be non-empty when editing an existing file"

        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            return "ERROR: old_text was not found; read the file again and include more context"
        if occurrences > 1:
            return "ERROR: old_text occurs more than once; provide more context"

        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Successfully edited {path}"

    def search_code(self, query: str) -> str:
        if not query:
            return "ERROR: search query cannot be empty"

        try:
            result = subprocess.run(
                [
                    "rg",
                    "-F",
                    "-n",
                    "-C",
                    "2",
                    "--hidden",
                    "--glob",
                    "!.git/**",
                    "--glob",
                    "!.env*",
                    "--glob",
                    "!**/.env*",
                    "--glob",
                    "!**/credentials*",
                    "--glob",
                    "!**/*.pem",
                    "--glob",
                    "!**/*.key",
                    "--glob",
                    "!**/*.p12",
                    "--glob",
                    "!**/*.pfx",
                    "--",
                    query,
                    ".",
                ],
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            return self._search_without_rg(query)
        except subprocess.TimeoutExpired:
            return "ERROR: search timed out after 30 seconds"

        if result.returncode not in (0, 1):
            return _truncate(result.stderr or "search failed")
        return _truncate(result.stdout or "(no matches)")

    def _search_without_rg(self, query: str) -> str:
        matches: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            if any(part in IGNORED_DIRECTORIES for part in relative.parts):
                continue
            if _is_secret_path(relative):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            matching_lines = {
                line_number
                for line_number, line in enumerate(lines, start=1)
                if query in line
            }
            context_lines = {
                line_number
                for match in matching_lines
                for line_number in range(max(1, match - 2), min(len(lines), match + 2) + 1)
            }
            for line_number in sorted(context_lines):
                marker = ":" if line_number in matching_lines else "-"
                matches.append(f"{relative}{marker}{line_number}{marker}{lines[line_number - 1]}")
        return _truncate("\n".join(matches) or "(no matches)")

    def _test_command(self) -> list[str] | None:
        configured = os.environ.get("TEST_COMMAND")
        if configured:
            try:
                command = shlex.split(configured)
            except ValueError as exc:
                raise ValueError(f"TEST_COMMAND is invalid: {exc}") from exc
            if not command:
                raise ValueError("TEST_COMMAND cannot be empty")
            return command

        if (self.root / "package.json").exists():
            return ["npm", "test"]
        if (self.root / "Cargo.toml").exists():
            return ["cargo", "test"]
        if (self.root / "go.mod").exists():
            return ["go", "test", "./..."]

        python_test_config = any(
            (self.root / name).exists()
            for name in ("pytest.ini", "tox.ini", "setup.cfg")
        )
        if python_test_config:
            return [sys.executable, "-m", "pytest", "-q"]

        tests_directory = self.root / "tests"
        if tests_directory.is_dir() and any(tests_directory.rglob("test*.py")):
            return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

        return None

    def run_tests(self) -> str:
        command = self._test_command()
        if command is None:
            return json.dumps(
                {
                    "status": "skipped",
                    "reason": "No supported test command detected. Set TEST_COMMAND to configure one.",
                }
            )

        test_environment = _test_environment()
        with tempfile.TemporaryDirectory(prefix="agent-test-home-") as test_home:
            test_environment.update(
                {
                    "HOME": test_home,
                    "USERPROFILE": test_home,
                    "XDG_CONFIG_HOME": os.path.join(test_home, ".config"),
                    "PYTHONNOUSERSITE": "1",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_TERMINAL_PROMPT": "0",
                    "NO_PROXY": "*",
                    "no_proxy": "*",
                }
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    text=True,
                    capture_output=True,
                    timeout=120,
                    env=test_environment,
                )
            except FileNotFoundError as exc:
                return json.dumps(
                    {
                        "status": "failed",
                        "command": command,
                        "output": f"Command not found: {exc.filename}",
                    }
                )
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or "") + (exc.stderr or "")
                return json.dumps(
                    {
                        "status": "failed",
                        "command": command,
                        "output": f"Timed out after 120 seconds\n{_truncate(output)}",
                    }
                )
        output = _truncate((result.stdout or "") + (result.stderr or ""))
        return json.dumps(
            {
                "status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "command": command,
                "output": output,
            }
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        handlers: dict[str, Callable[..., str]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
            "edit_file": self.edit_file,
            "run_tests": self.run_tests,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"TOOL ERROR: unknown tool {name}"
        try:
            return _truncate(str(handler(**arguments)))
        except Exception as exc:  # Return errors to the model so it can recover.
            return f"TOOL ERROR: {type(exc).__name__}: {exc}"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a focused line range from a UTF-8 text file using a repository-relative path. "
            "Use search_code first, then read only the relevant lines. Output is line-numbered "
            "for navigation; omit the numeric prefixes when using text in edit_file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_code",
        "description": (
            "Search the CONTENTS of repository files for literal text, a symbol, a function, "
            "a class, an error message, or behavior and return line numbers with nearby "
            "context. Do not use this tool to search for filenames; the repository file "
            "list is already provided. Omit the result's file and line prefixes when editing."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": (
            "Make one exact replacement in an existing UTF-8 file. "
            "Use an empty old_text only when creating a new file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_tests",
        "description": "Run the configured repository test command and return its exit status and output.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


SYSTEM_PROMPT = """
You are a careful software engineering agent fixing one GitHub issue.

You have a small number of model calls, so work quickly and reserve enough calls to
edit, test, and finish:
1. Use search_code to locate the relevant symbol or behavior. Search uses literal text
   and returns line numbers with nearby context.
2. Use read_file with start_line and end_line to inspect only the relevant section.
   Do not reread an entire large file from the beginning. The returned lines are
   prefixed with line numbers for navigation; omit those prefixes when copying
   old_text or new_text for edit_file.
3. Inspect relevant tests only if necessary, then make the smallest reasonable edit
   with edit_file. Once you have enough information to make a reasonable fix, edit
   instead of continuing to investigate.
4. Do not run the test suite before making an edit unless the issue specifically
   requires reproducing an existing failure.
5. Run the tests after editing. If they fail because of your change, make the smallest
   follow-up edit and run them again.
6. If tests pass and the issue is fixed, immediately return your final summary.
7. If the issue cannot be fixed, stop with a concrete explanation rather than exploring
   unrelated code.

Execution constraints:
- You should normally make the first edit within 4-5 model calls.
- Do not repeatedly search for code you have already located.
- Do not inspect unrelated initialization or entrypoint code unless required.
- Do not search broadly for tests when the repository already contains an obvious test file.
- Reserve at least two calls after the first edit for testing and finalizing.
- When changing a CLI, API, parser, or validation path, inspect downstream callers and
  update relevant tests before declaring the change complete. Declare each argument
  exactly once, and test both the new invocation and the existing normal invocation.

Rules:
- Treat the issue as a bug report, not as permission to make unrelated refactors.
- Use only the supplied repository tools.
- Do not create commits, branches, or pull requests; the harness does that.
- Never try to access files outside the repository.
- Do not claim tests passed unless run_tests reported that they passed.
- In your final response, summarize the change, tests, and any remaining limitation.
""".strip()


class Model(Protocol):
    def create(
        self,
        *,
        instructions: str,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        tool_choice: Any | None = None,
    ) -> Any:
        ...


class OpenAIModel:
    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentError(
                "The openai package is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        self.model = model
        self.client = OpenAI()

    def create(
        self,
        *,
        instructions: str,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        tool_choice: Any | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_items,
            "tools": tools,
        }
        if tool_choice is not None:
            request["tool_choice"] = {
                "type": "function",
                "name": tool_choice["function"]["name"],
            }
        return self.client.responses.create(**request)


class GeminiModel:
    """Gemini adapter using Google's OpenAI-compatible Chat Completions API."""

    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentError(
                "The openai package is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AgentError("GEMINI_API_KEY is not set")

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    @staticmethod
    def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Responses function definitions to Chat Completions format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _messages(instructions: str, input_items: list[Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions}
        ]

        for item in input_items:
            if not isinstance(item, dict):
                continue

            if item.get("role") == "user":
                messages.append(
                    {"role": "user", "content": item.get("content", "")}
                )
            elif item.get("type") == "assistant_message":
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": item.get("content") or None,
                }
                tool_calls = item.get("tool_calls") or []
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
            elif item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": item.get("output", ""),
                    }
                )

        return messages

    def create(
        self,
        *,
        instructions: str,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        tool_choice: Any | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(instructions, input_items),
            "tools": self._chat_tools(tools),
            "tool_choice": "auto" if tool_choice is None else tool_choice,
        }
        response = self.client.chat.completions.create(**request)
        if not response.choices:
            raise AgentError("Gemini returned no completion choices")

        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
        raw_tool_calls = getattr(message, "tool_calls", None) or []

        if not raw_tool_calls:
            return SimpleNamespace(
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                ],
                output_text=content,
            )

        normalized_tool_calls: list[dict[str, Any]] = []
        function_calls: list[dict[str, Any]] = []
        for raw_call in raw_tool_calls:
            function = raw_call.function
            call_id = raw_call.id
            normalized_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function.name,
                    "arguments": function.arguments,
                },
            }
            # Gemini 3 requires this encrypted signature to be returned in
            # the same assistant tool call on the next Chat Completions turn.
            extra_content = getattr(raw_call, "extra_content", None)
            if extra_content is None:
                model_dump = getattr(raw_call, "model_dump", None)
                if callable(model_dump):
                    dumped_call = model_dump()
                    if isinstance(dumped_call, dict):
                        extra_content = dumped_call.get("extra_content")
            if extra_content:
                if hasattr(extra_content, "model_dump"):
                    extra_content = extra_content.model_dump(exclude_none=True)
                normalized_call["extra_content"] = extra_content
            normalized_tool_calls.append(normalized_call)
            function_calls.append(
                {
                    "type": "function_call",
                    "name": function.name,
                    "arguments": function.arguments,
                    "call_id": call_id,
                }
            )

        return SimpleNamespace(
            output=[
                {
                    "type": "assistant_message",
                    "role": "assistant",
                    "content": content,
                    "tool_calls": normalized_tool_calls,
                },
                *function_calls,
            ],
            output_text=content,
        )


def create_model(provider: str, model: str) -> Model:
    if provider == "gemini":
        return GeminiModel(model)
    if provider == "openai":
        return OpenAIModel(model)
    raise AgentError(f"Unsupported LLM_PROVIDER: {provider}. Use gemini or openai.")


def mini_model_name(provider: str, model: str) -> str:
    """Return a LiteLLM model name with an explicit provider prefix."""
    if "/" in model:
        return model
    prefix = "gemini" if provider == "gemini" else "openai"
    return f"{prefix}/{model}"


def mini_docker_run_args(root: Path) -> list[str]:
    """Build the mini agent's isolated, repository-only Docker arguments."""
    return [
        "--rm",
        "--network",
        "none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev",
        "-v",
        f"{root.resolve()}:/workspace:rw",
    ]


def mini_command_makes_edit(command: str) -> bool:
    """Recognize common shell commands that can modify repository files."""
    patterns = (
        r"\bsed\s+-i\b",
        r"\bperl\s+-i\b",
        r"\b(?:apply_patch|patch)\b",
        r"\b(?:cp|mv|rm|tee|touch|mkdir)\b",
        r"(?:cat|echo|printf)\b[^\n;]*>{1,2}",
        r"\bpython(?:\d+(?:\.\d+)?)?\s+(?:-c|-)(?:\s|$).*"
        r"(?:open\(|write_text|write_bytes|\.write\()",
        r"\bgit\s+(?:apply|checkout|restore)\b",
    )
    return any(re.search(pattern, command, re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _mini_copy_ignore(path: str, names: list[str], root: Path) -> list[str]:
    ignored: list[str] = []
    source = Path(path)
    for name in names:
        relative = (source / name).relative_to(root)
        if (
            name == ".git"
            or any(part in MINI_IGNORED_DIRECTORIES for part in relative.parts)
            or _is_secret_path(relative)
        ):
            ignored.append(name)
    return ignored


def _mini_snapshot(root: Path) -> dict[str, bytes]:
    """Capture files that may be synchronized from the mini sandbox."""
    files: dict[str, bytes] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or any(
            part in MINI_IGNORED_DIRECTORIES for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise AgentError(
                f"The mini engine does not support symlinks in the checkout: {relative}"
            )
        if path.is_file():
            files[relative.as_posix()] = path.read_bytes()
    return files


def _reject_mini_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or any(
            part in MINI_IGNORED_DIRECTORIES for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise AgentError(
                f"The mini engine does not support symlinks in the checkout: {relative}"
            )


def _prepare_mini_sandbox(root: Path) -> tuple[Path, Path, dict[str, bytes]]:
    """Copy the safe part of a checkout into a temporary sandbox directory."""
    _reject_mini_symlinks(root)
    sandbox_directory = Path(
        tempfile.mkdtemp(prefix="mini-sandbox-", dir=tempfile.gettempdir())
    )
    sandbox_root = sandbox_directory / "workspace"
    try:
        shutil.copytree(
            root,
            sandbox_root,
            symlinks=True,
            ignore=lambda path, names: _mini_copy_ignore(path, names, root),
        )
        before = _mini_snapshot(sandbox_root)
    except Exception:
        shutil.rmtree(sandbox_directory, ignore_errors=True)
        raise
    return sandbox_directory, sandbox_root, before


def _sync_mini_sandbox(
    root: Path,
    sandbox_root: Path,
    before: dict[str, bytes],
) -> None:
    """Copy safe mini-agent changes back and reject protected-file changes."""
    after = _mini_snapshot(sandbox_root)
    changed = {
        relative
        for relative in set(before) | set(after)
        if before.get(relative) != after.get(relative)
    }
    protected = sorted(
        relative
        for relative in changed
        if _is_protected_path(Path(relative)) or _is_secret_path(Path(relative))
    )
    if protected:
        raise AgentError(
            "The mini agent changed protected workflow or credential paths; "
            "no pull request will be created:\n" + "\n".join(protected)
        )

    for relative in changed:
        target = root / relative
        if relative in after:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(after[relative])
        elif target.exists():
            target.unlink()


def run_mini_agent(
    root: Path,
    issue: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_steps: int,
    budget: RunBudget | None = None,
    log: Callable[[str], None] = print,
) -> AgentResult:
    """Run mini-SWE-agent as an optional, Docker-isolated inner agent.

    The outer harness still owns branch creation, validation, testing, commits,
    pushes, and pull requests. The mini agent only edits a temporary checkout copy.
    """
    if budget is not None:
        max_steps = min(max_steps, budget.remaining)
    if max_steps <= 0:
        raise AgentError("No model-call budget remains for the mini engine.")
    if shutil.which("docker") is None:
        raise AgentError(
            "The mini engine requires Docker. Install Docker and build the sandbox "
            "image with: docker build -f Dockerfile.agent -t agent-fix-sandbox:latest ."
        )

    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        raise AgentError("GEMINI_API_KEY is not set")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise AgentError("OPENAI_API_KEY is not set")

    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel
    except ImportError as exc:
        raise AgentError(
            "The mini engine is not installed. Run: "
            "python -m pip install -r requirements-mini.txt"
        ) from exc

    class FocusedMiniAgent(DefaultAgent):
        """Bound unproductive exploration without forbidding targeted reads."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.edit_made = False
            self.no_edit_calls_after_deadline = 0
            super().__init__(*args, **kwargs)

        def query(self) -> dict[str, Any]:
            if not self.edit_made and self.n_calls >= MINI_EDIT_DEADLINE_CALL - 1:
                if self.n_calls >= MINI_EDIT_DEADLINE_CALL:
                    message = (
                        "Implementation checkpoint reached. A targeted read is still allowed "
                        "if needed, but do not repeat exploration. Make the smallest edit "
                        "as soon as you have enough context."
                    )
                else:
                    message = (
                        "You have used most of the investigation budget. Make the smallest "
                        "reasonable edit on this call, then run tests; do not keep exploring."
                    )
                self.add_messages(self.model.format_message(role="user", content=message))
            return super().query()

        def execute_actions(self, message: dict) -> list[dict]:
            outputs: list[dict[str, Any]] = []
            for action in message.get("extra", {}).get("actions", []):
                command = str(action.get("command", ""))
                log(f"Mini step {self.n_calls}: bash {_truncate(command, 500)}")
                output = self.env.execute(action)
                outputs.append(output)
                if output.get("returncode") == 0 and (
                    mini_command_makes_edit(command)
                    or _mini_snapshot(sandbox_root) != sandbox_before
                ):
                    self.edit_made = True
                    self.no_edit_calls_after_deadline = 0
                elif (
                    not self.edit_made
                    and self.n_calls >= MINI_EDIT_DEADLINE_CALL
                ):
                    self.no_edit_calls_after_deadline += 1
                    log(
                        "Mini progress checkpoint: "
                        f"{self.no_edit_calls_after_deadline}/{MINI_STALL_CALLS} "
                        "post-deadline calls without an edit"
                    )

            observation_messages = self.add_messages(
                *self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
            )
            if (
                not self.edit_made
                and self.no_edit_calls_after_deadline >= MINI_STALL_CALLS
            ):
                raise MiniEditDeadlineExceeded(
                    "The mini agent made no implementation progress after its checkpoint."
                )
            return observation_messages

    image = os.environ.get("MINI_DOCKER_IMAGE", MINI_DEFAULT_IMAGE)
    try:
        cost_limit = float(
            os.environ.get("MINI_COST_LIMIT", str(MINI_DEFAULT_COST_LIMIT))
        )
    except ValueError as exc:
        raise AgentError("MINI_COST_LIMIT must be a number") from exc

    mini_model = LitellmModel(
        model_name=mini_model_name(provider, model),
        model_kwargs={"temperature": 0},
    )
    sandbox_directory, sandbox_root, sandbox_before = _prepare_mini_sandbox(root)
    try:
        environment = DockerEnvironment(
            image=image,
            cwd="/workspace",
            env={
                "CI": "true",
                "HOME": "/tmp/agent-home",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
            # Never forward GH_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY, or any other
            # host variable into the shell that the model controls.
            forward_env=[],
            run_args=mini_docker_run_args(sandbox_root),
            timeout=120,
            container_timeout="2h",
        )
    except Exception as exc:
        shutil.rmtree(sandbox_directory, ignore_errors=True)
        raise AgentError(
            f"Could not start the mini Docker sandbox using image {image}: {exc}"
        ) from exc

    system_template = """
You are a focused software engineer working in /workspace.

Solve the GitHub issue with the smallest correct change. You may inspect, edit,
and test files in /workspace using bash. Work directly on the provided checkout
copy. Do not use git, commit, push, open pull requests, access the network, or
inspect credentials. Do not modify .github/workflows, .github/actions, .env files,
or private-key files. Run the repository's tests after editing.
""".strip()
    instance_template = """
Please solve this issue: {{ task }}

You can execute bash commands and edit files to implement the necessary changes.

## Required workflow

1. Use the repository map and inspect only files relevant to the issue.
2. Make the smallest reasonable implementation edit after you have enough evidence;
   do not spend the whole budget rereading the same code.
3. Run the repository tests after editing and fix failures caused by your change.
4. Stop exploring once the issue is fixed and the tests pass.
5. Finish with exactly one bash tool call whose command is:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine that command with another command.

## Command execution rules

Every response must contain exactly one bash tool call. Use the bash tool with a
command argument, for example:

```json
{"command": "rg -n 'pattern' file.py"}
```

The shell runs in a fresh process for each call, but file changes persist. Do not
spend the whole budget reading files: after the relevant code is located, edit it.
""".strip()
    task = (
        f"GitHub issue #{issue['number']}\n"
        f"Title: {issue.get('title', '')}\n\n"
        f"Description:\n{issue.get('body') or '(no issue description)'}\n\n"
        "Repository map:\n"
        f"{build_repository_index(root).render(issue.get('title', '') + ' ' + (issue.get('body') or ''))}\n\n"
        "Make the implementation change in the current checkout, verify it with tests, "
        "and then submit the final summary command."
    )

    trajectory_directory = Path(
        tempfile.mkdtemp(prefix="mini-swe-agent-", dir=tempfile.gettempdir())
    )
    trajectory_path = trajectory_directory / "trajectory.json"
    agent = FocusedMiniAgent(
        mini_model,
        environment,
        system_template=system_template,
        instance_template=instance_template,
        step_limit=max_steps,
        cost_limit=cost_limit,
        wall_time_limit_seconds=600,
        output_path=trajectory_path,
    )
    log(
        f"Running mini-SWE-agent with model {mini_model_name(provider, model)} "
        f"in Docker image {image}"
    )
    outcome: dict[str, Any] | None = None
    run_error: Exception | None = None
    stopped_without_edit = False
    try:
        outcome = agent.run(task)
    except MiniEditDeadlineExceeded as exc:
        stopped_without_edit = True
        log(f"Mini agent stopped early: {exc}")
    except Exception as exc:
        run_error = exc
    finally:
        environment.cleanup()

    try:
        _sync_mini_sandbox(root, sandbox_root, sandbox_before)
    finally:
        shutil.rmtree(trajectory_directory, ignore_errors=True)
        shutil.rmtree(sandbox_directory, ignore_errors=True)

    if budget is not None:
        budget.consume(agent.n_calls, phase="mini")

    if run_error is not None:
        raise AgentError(f"mini-SWE-agent failed: {run_error}") from run_error
    if stopped_without_edit:
        return AgentResult(
            f"mini-SWE-agent stopped after {agent.n_calls} model calls without making "
            "an edit; the harness will retry with its focused tool agent.",
            agent.n_calls,
        )
    if outcome is None:
        raise AgentError("mini-SWE-agent returned no result")

    status = str(outcome.get("exit_status") or "unknown")
    submission = str(outcome.get("submission") or "").strip()
    if submission:
        summary = submission
    else:
        changed = _run_process(
            ["git", "diff", "--stat"], cwd=root, check=False
        ).strip()
        summary = (
            f"mini-SWE-agent stopped with status {status} after {agent.n_calls} "
            "model calls.\n\n"
            f"Changed files:\n{changed or '(no tracked-file diff reported)'}\n\n"
            "The harness will run the final tests before creating a pull request."
        )
    return AgentResult(summary, agent.n_calls)


@dataclass
class AgentResult:
    final_message: str
    steps: int


class AgentRunner:
    def __init__(
        self,
        workspace: RepoWorkspace,
        model: Model,
        *,
        max_steps: int = MAX_STEPS,
        budget: RunBudget | None = None,
        request_delay: float = 0.0,
        log: Callable[[str], None] = print,
    ):
        self.workspace = workspace
        self.model = model
        self.max_steps = max_steps
        self.budget = budget
        self.request_delay = max(0.0, request_delay)
        self.log = log

    def _has_worktree_changes(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.workspace.root,
                text=True,
                capture_output=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _change_summary(self) -> str:
        status = _run_process(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.workspace.root,
            check=False,
        ).strip()
        stat = _run_process(
            ["git", "diff", "--stat"], cwd=self.workspace.root, check=False
        ).strip()
        diff = _run_process(
            ["git", "diff", "--unified=0"], cwd=self.workspace.root, check=False
        )
        changed_lines = [
            line
            for line in diff.splitlines()
            if (line.startswith("+") or line.startswith("-") or line.startswith("@@"))
            and not line.startswith(("+++", "---"))
        ]
        details = _truncate("\n".join(changed_lines), 4_000)
        parts = ["Changed files:", status or "(unable to list changed files)"]
        if stat:
            parts.extend(["", "Diff statistics:", stat])
        if details:
            parts.extend(["", "Changed lines:", details])
        return "\n".join(parts)

    def run(self, issue: dict[str, Any]) -> AgentResult:
        issue_number = issue["number"]
        title = issue.get("title", "")
        body = issue.get("body") or "(no issue description)"
        repository_map = build_repository_index(self.workspace.root).render(
            f"{title} {body}"
        )
        initial_prompt = (
            f"GitHub issue #{issue_number}\n"
            f"Title: {title}\n\n"
            f"Description:\n{body}\n\n"
            f"Repository map:\n{repository_map}\n\n"
            "Fix this issue in the current repository."
        )
        input_items: list[Any] = [
            {
                "role": "user",
                "content": initial_prompt,
            }
        ]
        budget = self.budget or RunBudget(self.max_steps)
        search_calls = 0
        edit_made = False
        phase = "investigate"
        recovery_turn = False
        observations: list[str] = []
        implementation_context_compacted = False
        seen_actions: set[tuple[str, str]] = set()
        stale_actions = 0
        last_step = 0

        step_limit = self.max_steps
        step_limit = min(step_limit, budget.remaining)

        for step in range(1, step_limit + 1):
            while budget.phase_remaining(phase) <= 0:
                if phase == "investigate":
                    phase = "implement"
                    recovery_turn = False
                elif phase == "implement" and not edit_made:
                    phase = "repair"
                    recovery_turn = True
                else:
                    break
            if budget.phase_remaining(phase) <= 0:
                break

            last_step = step
            self.log(f"\n--- Agent step {step} ---")
            if step > 1 and self.request_delay:
                time.sleep(self.request_delay)

            if phase == "investigate":
                allowed_tools = [
                    tool for tool in TOOLS if tool["name"] != "run_tests"
                ]
                if search_calls >= 2:
                    allowed_tools = [
                        tool
                        for tool in allowed_tools
                        if tool["name"] != "search_code"
                    ]
            elif phase == "implement":
                allowed_tools = [
                    tool for tool in TOOLS if tool["name"] == "edit_file"
                ]
            elif phase == "repair" and recovery_turn:
                allowed_tools = [
                    tool
                    for tool in TOOLS
                    if tool["name"] in {"read_file", "edit_file"}
                ]
            elif phase == "repair":
                allowed_tools = [
                    tool for tool in TOOLS if tool["name"] == "edit_file"
                ]
            else:
                allowed_tools = [
                    tool
                    for tool in TOOLS
                    if tool["name"] in {"read_file", "edit_file", "run_tests"}
                ]
            allowed_tool_names = {tool["name"] for tool in allowed_tools}
            force_edit = phase == "implement" or (
                phase == "repair" and not recovery_turn
            )
            tool_choice = (
                {
                    "type": "function",
                    "function": {"name": "edit_file"},
                }
                if force_edit
                else None
            )

            if phase == "implement" and not implementation_context_compacted:
                evidence = _truncate(
                    "\n\n".join(observations), MAX_TOOL_OUTPUT
                )
                input_items = [
                    {
                        "role": "user",
                        "content": (
                            f"{initial_prompt}\n\n"
                            "Relevant evidence collected during localization:\n"
                            f"{evidence or '(No successful read was collected; use the map.)'}\n\n"
                            "The implementation phase has started. Apply the smallest correct edit."
                        ),
                    }
                ]
                implementation_context_compacted = True

            remaining = min(self.max_steps - step + 1, budget.remaining)
            turn_instructions = (
                f"{SYSTEM_PROMPT}\n\n"
                f"PHASE: {phase.upper()}\n"
                f"CURRENT MODEL CALL: {step} of {self.max_steps}\n"
                f"GLOBAL CALLS REMAINING INCLUDING THIS ONE: {remaining}\n"
                f"PHASE CALLS REMAINING: {budget.phase_remaining(phase)}\n"
                f"SEARCH CALLS USED: {search_calls} of 2\n"
            )
            if search_calls >= 2:
                turn_instructions += (
                    "You have exhausted your search budget. Do not continue exploring "
                    "the repository; use the information already gathered.\n"
                )
            if phase == "investigate":
                turn_instructions += (
                    "Investigate only enough to identify the likely root cause and "
                    "target edit. Repeated observations will be rejected.\n"
                )
            elif phase == "implement":
                turn_instructions += (
                    "Investigation is complete for this run. "
                    "Call edit_file now. Do not call a read or search tool.\n"
                )
            elif phase == "repair" and recovery_turn:
                turn_instructions += (
                    "The previous edit attempt did not apply. Make one "
                    "targeted read_file call to correct the exact replacement, "
                    "then call edit_file. Do not resume general exploration.\n"
                )
            elif phase == "repair":
                turn_instructions += (
                    "Repair the current patch with edit_file now. Do not read or "
                    "search again.\n"
                )
            else:
                turn_instructions += (
                    "Verify the change with run_tests. If a test fails, make a "
                    "focused repair and rerun it; otherwise finish with a summary.\n"
                )
            if remaining <= 3:
                turn_instructions += (
                    "Use the remaining calls only for implementation, testing, or "
                    "finalizing; do not start new exploration.\n"
                )

            for attempt in range(2):
                try:
                    budget.consume(phase=phase)
                    request: dict[str, Any] = {
                        "instructions": turn_instructions,
                        "input_items": input_items,
                        "tools": allowed_tools,
                    }
                    if tool_choice is not None:
                        request["tool_choice"] = tool_choice
                    response = self.model.create(**request)
                    break
                except Exception as exc:
                    retry_delay = _rate_limit_delay(exc)
                    if retry_delay is None or attempt == 1:
                        raise
                    self.log(
                        f"Rate limited; waiting {retry_delay:.1f}s before retrying"
                    )
                    time.sleep(retry_delay)
            output_items = list(_item_value(response, "output", []) or [])
            input_items.extend(output_items)

            tool_calls = [
                item for item in output_items if _item_value(item, "type") == "function_call"
            ]
            if not tool_calls:
                if force_edit:
                    if phase == "implement":
                        phase = "repair"
                        recovery_turn = True
                    else:
                        recovery_turn = False
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "The implementation is not complete. Use the available "
                                "edit_file tool on your next turn."
                            ),
                        }
                    )
                    continue
                if recovery_turn:
                    recovery_turn = False
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "Do not finish without applying the fix. Call edit_file "
                                "on your next turn."
                            ),
                        }
                    )
                    continue
                if phase == "investigate":
                    phase = "implement"
                    continue
                final_message = _item_value(response, "output_text", "") or "Agent finished without a summary."
                return AgentResult(str(final_message), step)

            edit_succeeded = False
            test_failure_transition = False
            for call in tool_calls:
                name = _item_value(call, "name")
                raw_arguments = _item_value(call, "arguments", "{}")
                call_id = _item_value(call, "call_id")
                self.log(f"Tool: {name}({raw_arguments})")
                if name not in allowed_tool_names:
                    result = (
                        f"TOOL ERROR: {name} is unavailable in the current phase; "
                        "use one of the available tools."
                    )
                else:
                    if name == "search_code":
                        search_calls += 1
                    try:
                        arguments = json.loads(raw_arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("tool arguments must be a JSON object")
                        action_key = (
                            name,
                            json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                        )
                        if name in {"search_code", "read_file"} and action_key in seen_actions:
                            result = (
                                "ALREADY_OBSERVED: this exact search or read was "
                                "already performed. Use its existing result or edit."
                            )
                            stale_actions += 1
                        else:
                            if name in {"search_code", "read_file"}:
                                seen_actions.add(action_key)
                            result = self.workspace.call_tool(name, arguments)
                            if name in {"search_code", "read_file"}:
                                stale_actions = 0
                    except Exception as exc:
                        result = f"TOOL ERROR: {type(exc).__name__}: {exc}"
                    if name == "edit_file" and result.startswith(
                        ("Successfully edited", "Successfully created")
                    ):
                        edit_made = True
                        edit_succeeded = True
                    elif name in {"search_code", "read_file"} and not result.startswith(
                        ("ERROR:", "TOOL ERROR:")
                    ):
                        observations.append(f"{name}:\n{_truncate(result, 4_000)}")
                    elif name == "run_tests":
                        try:
                            if json.loads(result).get("status") == "failed":
                                phase = "repair"
                                recovery_turn = True
                                test_failure_transition = True
                        except (TypeError, json.JSONDecodeError):
                            pass
                self.log(_truncate(result, 1_000))
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )

            if edit_succeeded:
                phase = "verify"
                recovery_turn = False
            elif force_edit and phase == "implement":
                phase = "repair"
                recovery_turn = True
            elif force_edit and phase == "repair":
                recovery_turn = False
            elif recovery_turn and not test_failure_transition:
                recovery_turn = False
            elif phase == "investigate" and stale_actions >= 2:
                phase = "implement"

        if self._has_worktree_changes():
            message = (
                f"Agent finished its phase budget after making "
                "a change without returning a final summary.\n\n"
                f"{self._change_summary()}\n\n"
                "The harness will run the final tests and prepare the pull request "
                "if they pass."
            )
            self.log(message)
            return AgentResult(message, last_step)

        raise AgentError(
            "Agent exhausted its bounded phase budgets without making a change"
        )


def repository_root(start: Path) -> Path:
    output = _run_process(
        ["git", "rev-parse", "--show-toplevel"], cwd=start.resolve(), check=True
    ).strip()
    if not output:
        raise AgentError("The current directory is not inside a Git repository")
    return Path(output).resolve()


def require_clean_worktree(root: Path) -> None:
    status = _run_process(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root
    ).strip()
    if status:
        raise AgentError(
            "The working tree must be clean before the agent starts.\n" + status
        )


def _changed_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for command in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        output = _run_process(command, cwd=root, check=False)
        paths.update(line.strip() for line in output.splitlines() if line.strip())
    return paths


def validate_agent_changes(root: Path) -> None:
    protected = sorted(
        path for path in _changed_paths(root) if _is_protected_path(Path(path))
    )
    if protected:
        raise AgentError(
            "The agent changed protected workflow or credential paths; "
            "no pull request will be created:\n" + "\n".join(protected)
        )


def has_worktree_changes(root: Path) -> bool:
    """Return whether the agent produced tracked or untracked worktree changes."""
    status = _run_process(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=False,
    )
    return bool(status.strip())


def get_issue(root: Path, issue_number: int) -> dict[str, Any]:
    raw = _run_process(
        [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--json",
            "number,title,body,url",
        ],
        cwd=root,
    )
    try:
        issue = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentError(f"gh returned invalid issue JSON: {exc}") from exc
    if not issue.get("number") or not issue.get("title"):
        raise AgentError("gh returned an issue without a number or title")
    return issue


def find_open_issue_pr(root: Path, issue_number: int) -> str | None:
    raw = _run_process(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,body,url",
        ],
        cwd=root,
    )
    try:
        pull_requests = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentError(f"gh returned invalid pull request JSON: {exc}") from exc

    marker = re.compile(rf"^\s*Fixes\s+#{re.escape(str(issue_number))}\b", re.IGNORECASE | re.MULTILINE)
    title_marker = re.compile(rf"^Fix\s+#{re.escape(str(issue_number))}:\s*", re.IGNORECASE)
    for pull_request in pull_requests:
        title = str(pull_request.get("title") or "")
        body = str(pull_request.get("body") or "")
        if marker.search(body) or title_marker.search(title):
            return str(pull_request.get("url") or pull_request.get("number"))
    return None


def _remote_branch_exists(root: Path, branch: str) -> bool:
    output = _run_process(
        ["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
        cwd=root,
        check=False,
    )
    return any(line.rstrip().endswith(f"refs/heads/{branch}") for line in output.splitlines())


def create_branch(root: Path, issue_number: int) -> str:
    base_branch = f"agent/issue-{issue_number}"
    branch = base_branch
    existing = _run_process(["git", "branch", "--list", branch], cwd=root).strip()
    if existing or _remote_branch_exists(root, branch):
        suffix = os.environ.get("GITHUB_RUN_ID") or str(time.time_ns())
        branch = f"{base_branch}-retry-{suffix}"
        if _run_process(["git", "branch", "--list", branch], cwd=root).strip() or _remote_branch_exists(root, branch):
            raise AgentError(
                f"A retry branch already exists for issue #{issue_number}: {branch}"
            )
    _run_process(["git", "switch", "--create", branch], cwd=root)
    return branch


def final_test_status(workspace: RepoWorkspace) -> dict[str, Any]:
    raw = workspace.run_tests()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "failed", "output": raw}
    if result.get("status") != "passed":
        raise AgentError(
            "The final test run did not pass; no commit or pull request was created.\n"
            + str(result.get("output", ""))
            + str(result.get("reason", ""))
        )
    return result


def create_pr(root: Path, issue: dict[str, Any], branch: str, summary: str) -> str:
    validate_agent_changes(root)
    _run_process(["git", "add", "--all"], cwd=root)
    staged = _run_process(["git", "diff", "--cached", "--stat"], cwd=root).strip()
    if not staged:
        raise AgentError("The agent made no changes; no pull request was created.")

    _run_process(
        ["git", "commit", "-m", f"Fix issue #{issue['number']}"], cwd=root
    )
    _run_process(["git", "push", "--set-upstream", "origin", branch], cwd=root)

    body = (
        f"Automated fix for #{issue['number']}.\n\n"
        f"Fixes #{issue['number']}\n\n"
        "Agent summary:\n"
        f"{_truncate(summary, 8_000)}\n"
    )
    return _run_process(
        [
            "gh",
            "pr",
            "create",
            "--head",
            branch,
            "--title",
            f"Fix #{issue['number']}: {issue['title']}",
            "--body",
            body,
        ],
        cwd=root,
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    engine = os.environ.get("AGENT_ENGINE", DEFAULT_ENGINE).lower()
    if engine not in {"custom", "mini"}:
        engine = DEFAULT_ENGINE
    if provider == "openai":
        default_model = os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    else:
        default_model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version="GitHub issue agent MVP",
    )
    parser.add_argument("issue_number", type=int, nargs="?", help="GitHub issue number to fix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the agent and tests, then print the diff without committing, pushing, or opening a PR",
    )
    parser.add_argument(
        "--engine",
        default=engine,
        choices=["custom", "mini"],
        help="Agent engine (default: $AGENT_ENGINE or custom)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum model turns (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--provider",
        default=provider,
        choices=["gemini", "openai"],
        help="LLM provider (default: $LLM_PROVIDER or gemini)",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Model name (default depends on the provider)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print selected provider, model, and maximum step count",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        print(f"Engine: {args.engine}")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print(f"Max steps: {args.max_steps}")
    if args.issue_number is None:
        parser = build_parser()
        parser.print_usage()
        return 1
    if args.issue_number <= 0:
        raise AgentError("issue_number must be positive")
    if args.max_steps <= 0:
        raise AgentError("max_steps must be positive")

    budget = RunBudget(args.max_steps)

    root = repository_root(Path.cwd())
    require_clean_worktree(root)
    issue = get_issue(root, args.issue_number)
    print(f"Working on #{issue['number']}: {issue['title']}")

    existing_pr = find_open_issue_pr(root, args.issue_number)
    if existing_pr:
        print(f"An open pull request already exists for this issue: {existing_pr}")
        return 0

    branch = "(dry run)"
    if not args.dry_run:
        branch = create_branch(root, args.issue_number)
        print(f"Created branch {branch}")

    workspace = RepoWorkspace(root)
    if args.engine == "mini":
        result = run_mini_agent(
            root,
            issue,
            provider=args.provider,
            model=args.model,
            max_steps=min(budget.remaining, MINI_MAX_CALLS),
            budget=budget,
        )
        if not has_worktree_changes(root):
            if not MINI_FALLBACK_ENABLED:
                raise AgentError(
                    "mini-SWE-agent made no edit; the fallback tool agent is disabled."
                )
            if budget.remaining <= 0:
                raise AgentError(
                    "The global model-call budget was exhausted without an edit; "
                    "no pull request was created."
                )
            print(
                f"\nMini agent made no edit. Retrying with the focused tool agent "
                f"using the {budget.remaining} remaining model calls."
            )
            runner = AgentRunner(
                workspace,
                create_model(args.provider, args.model),
                max_steps=budget.remaining,
                budget=budget,
                request_delay=(
                    float(os.environ.get("LLM_REQUEST_DELAY", GEMINI_REQUEST_DELAY_SECONDS))
                    if args.provider == "gemini"
                    else 0.0
                ),
            )
            result = runner.run(issue)
    else:
        runner = AgentRunner(
            workspace,
            create_model(args.provider, args.model),
            max_steps=args.max_steps,
            budget=budget,
            request_delay=(
                float(os.environ.get("LLM_REQUEST_DELAY", GEMINI_REQUEST_DELAY_SECONDS))
                if args.provider == "gemini"
                else 0.0
            ),
        )
        result = runner.run(issue)

    print("\nAgent finished:")
    print(result.final_message)
    validate_agent_changes(root)
    test_result = final_test_status(workspace)
    print(f"\nFinal tests: {json.dumps(test_result)}")

    if args.dry_run:
        status = _run_process(
            ["git", "status", "--short", "--untracked-files=all"], cwd=root, check=False
        ).strip()
        diff = _run_process(["git", "diff"], cwd=root, check=False)
        print("\nDry-run status:")
        print(status or "(no changes)")
        print("\nDry-run diff:")
        print(_truncate(diff, 30_000) if diff else "(no tracked-file diff; see status for new files)")
        return 0

    print("\nCreating pull request...")
    final_command = " ".join(str(part) for part in test_result.get("command", []))
    pr_summary = (
        f"{result.final_message}\n\n"
        f"Final validation passed: `{final_command or 'configured test command'}`"
    )
    pr_url = create_pr(root, issue, branch, pr_summary)
    print(f"Pull request created: {pr_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AgentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

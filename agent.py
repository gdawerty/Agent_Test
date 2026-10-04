#!/usr/bin/env python3
"""Small GitHub issue-to-PR coding agent.

The model can inspect and edit the checkout through a deliberately small tool
surface. Git and GitHub side effects stay in this process so the model cannot
commit, push, or open a pull request by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
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
MAX_STEPS = 15
MAX_READ_LINES = 200
GEMINI_REQUEST_DELAY_SECONDS = 5.0
CUSTOM_EDIT_FORCE_CALL = 7
MINI_DEFAULT_IMAGE = "agent-fix-sandbox:latest"
MINI_DEFAULT_COST_LIMIT = 3.0
MINI_EDIT_DEADLINE_CALL = 7
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
    """The mini engine exhausted its investigation window without editing."""


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

You have a small number of tool turns, so work quickly and reserve enough turns to
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
- You should normally make the first edit within 4-5 model turns.
- Do not repeatedly search for code you have already located.
- Do not inspect unrelated initialization or entrypoint code unless required.
- Do not search broadly for tests when the repository already contains an obvious test file.
- Reserve at least two turns after the first edit for testing and finalizing.
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
            if (
                isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and isinstance(tool_choice.get("function"), dict)
            ):
                request["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice["function"]["name"],
                }
            else:
                request["tool_choice"] = tool_choice
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
    log: Callable[[str], None] = print,
) -> AgentResult:
    """Run mini-SWE-agent as an optional, Docker-isolated inner agent.

    The outer harness still owns branch creation, validation, testing, commits,
    pushes, and pull requests. The mini agent only edits a temporary checkout copy.
    """
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
        """Add a hard implementation deadline to mini's bash loop."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.edit_made = False
            super().__init__(*args, **kwargs)

        def query(self) -> dict[str, Any]:
            if not self.edit_made and self.n_calls >= MINI_EDIT_DEADLINE_CALL - 1:
                if self.n_calls >= MINI_EDIT_DEADLINE_CALL:
                    message = (
                        "Implementation deadline reached. Read-only commands are blocked "
                        "until you edit a source or test file. Issue one edit command now."
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
            blocked_read_only_command = False
            for action in message.get("extra", {}).get("actions", []):
                command = str(action.get("command", ""))
                log(f"Mini step {self.n_calls}: bash {_truncate(command, 500)}")
                if (
                    not self.edit_made
                    and self.n_calls >= MINI_EDIT_DEADLINE_CALL
                    and not mini_command_makes_edit(command)
                ):
                    log("Mini command blocked until the agent edits a file")
                    blocked_read_only_command = True
                    outputs.append(
                        {
                            "output": (
                                "Blocked: you have reached the implementation deadline. "
                                "Use one shell command that edits the source or test file now."
                            ),
                            "returncode": 1,
                            "exception_info": "",
                        }
                    )
                    continue

                output = self.env.execute(action)
                outputs.append(output)
                if output.get("returncode") == 0 and (
                    mini_command_makes_edit(command)
                    or _mini_snapshot(sandbox_root) != sandbox_before
                ):
                    self.edit_made = True

            observation_messages = self.add_messages(
                *self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
            )
            if blocked_read_only_command and not self.edit_made:
                raise MiniEditDeadlineExceeded(
                    "The mini agent reached the implementation deadline without editing."
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

1. Inspect only the files relevant to the issue.
2. Make the smallest reasonable implementation edit by model call 7.
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
        request_delay: float = 0.0,
        log: Callable[[str], None] = print,
    ):
        self.workspace = workspace
        self.model = model
        self.max_steps = max_steps
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
        repository_files = _truncate(self.workspace.list_files(), MAX_TOOL_OUTPUT)
        input_items: list[Any] = [
            {
                "role": "user",
                "content": (
                    f"GitHub issue #{issue_number}\n"
                    f"Title: {title}\n\n"
                    f"Description:\n{body}\n\n"
                    f"Repository files:\n{repository_files}\n\n"
                    "Fix this issue in the current repository."
                ),
            }
        ]
        search_calls = 0
        edit_made = False

        for step in range(1, self.max_steps + 1):
            self.log(f"\n--- Agent step {step} ---")
            if step > 1 and self.request_delay:
                time.sleep(self.request_delay)

            allowed_tools = TOOLS
            if edit_made:
                allowed_tools = [
                    tool
                    for tool in TOOLS
                    if tool["name"] in {"read_file", "edit_file", "run_tests"}
                ]
            elif search_calls >= 2:
                allowed_tools = [
                    tool for tool in TOOLS if tool["name"] != "search_code"
                ]
            if step >= 6 and not edit_made:
                allowed_tools = [
                    tool
                    for tool in allowed_tools
                    if tool["name"] in {"read_file", "edit_file"}
                ]
            if step >= 7 and not edit_made:
                allowed_tools = [
                    tool for tool in allowed_tools if tool["name"] == "edit_file"
                ]
            allowed_tool_names = {tool["name"] for tool in allowed_tools}
            force_edit = step >= CUSTOM_EDIT_FORCE_CALL and not edit_made
            tool_choice = (
                {
                    "type": "function",
                    "function": {"name": "edit_file"},
                }
                if force_edit
                else None
            )

            remaining = self.max_steps - step + 1
            turn_instructions = (
                f"{SYSTEM_PROMPT}\n\n"
                f"CURRENT TURN: {step} of {self.max_steps}\n"
                f"TURNS REMAINING INCLUDING THIS ONE: {remaining}\n"
                f"SEARCH CALLS USED: {search_calls} of 2\n"
            )
            if search_calls >= 2:
                turn_instructions += (
                    "You have exhausted your search budget. Do not continue exploring "
                    "the repository; use the information already gathered.\n"
                )
            if step >= 6 and not edit_made:
                turn_instructions += (
                    "You must make an edit now unless you are genuinely blocked. "
                    "Only read_file and edit_file are available in this phase.\n"
                )
            if step >= 7 and not edit_made:
                turn_instructions += (
                    "You have not edited any code yet. Make the smallest reasonable "
                    "edit now; edit_file is the only available tool.\n"
                )
            if remaining <= 3:
                turn_instructions += (
                    "Stop exploring. Edit, test, and finalize now; do not make another "
                    "unnecessary search or read.\n"
                )

            for attempt in range(2):
                try:
                    response = self.model.create(
                        instructions=turn_instructions,
                        input_items=input_items,
                        tools=allowed_tools,
                        tool_choice=tool_choice,
                    )
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
                if force_edit and not edit_made:
                    raise AgentError(
                        "The model did not produce the required edit_file call "
                        f"by step {step}"
                    )
                final_message = _item_value(response, "output_text", "") or "Agent finished without a summary."
                return AgentResult(str(final_message), step)

            for call in tool_calls:
                name = _item_value(call, "name")
                raw_arguments = _item_value(call, "arguments", "{}")
                call_id = _item_value(call, "call_id")
                self.log(f"Tool: {name}({raw_arguments})")
                if name not in allowed_tool_names:
                    if force_edit:
                        raise AgentError(
                            f"The model requested {name} after the edit deadline; "
                            "the only permitted tool is edit_file"
                        )
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
                        result = self.workspace.call_tool(name, arguments)
                    except Exception as exc:
                        result = f"TOOL ERROR: {type(exc).__name__}: {exc}"
                    if name == "edit_file" and result.startswith(
                        ("Successfully edited", "Successfully created")
                    ):
                        edit_made = True
                self.log(_truncate(result, 1_000))
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )

        if self._has_worktree_changes():
            message = (
                f"Agent reached the maximum of {self.max_steps} steps after making "
                "a change without returning a final summary.\n\n"
                f"{self._change_summary()}\n\n"
                "The harness will run the final tests and prepare the pull request "
                "if they pass."
            )
            self.log(message)
            return AgentResult(message, self.max_steps)

        raise AgentError(f"Agent exceeded the maximum of {self.max_steps} steps without making a change")


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
            max_steps=args.max_steps,
        )
        if not has_worktree_changes(root):
            print(
                "\nMini agent made no edit. Retrying with the focused tool agent "
                "so this issue does not consume the remaining mini steps."
            )
            runner = AgentRunner(
                workspace,
                create_model(args.provider, args.model),
                max_steps=args.max_steps,
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

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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, Sequence


DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-5.6"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_TOOL_OUTPUT = 8_000
MAX_FILE_OUTPUT = 8_000
MAX_STEPS = 12
MAX_READ_LINES = 200
GEMINI_REQUEST_DELAY_SECONDS = 5.0
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


class AgentError(RuntimeError):
    """An expected failure that should be shown as a user-facing CLI error."""


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

        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=120,
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
            "Search repository text for literal text or a code symbol and return line numbers "
            "with nearby context. Omit the result's file and line prefixes when editing."
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
  update relevant tests before declaring the change complete.

Rules:
- Treat the issue as a bug report, not as permission to make unrelated refactors.
- Use only the supplied repository tools.
- Do not create commits, branches, or pull requests; the harness does that.
- Never try to access files outside the repository.
- Do not claim tests passed unless run_tests reported that they passed.
- In your final response, summarize the change, tests, and any remaining limitation.
""".strip()


class Model(Protocol):
    def create(self, *, instructions: str, input_items: list[Any], tools: list[dict[str, Any]]) -> Any:
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

    def create(self, *, instructions: str, input_items: list[Any], tools: list[dict[str, Any]]) -> Any:
        return self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
        )


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
    ) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(instructions, input_items),
            tools=self._chat_tools(tools),
            tool_choice="auto",
        )
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

        for step in range(1, self.max_steps + 1):
            self.log(f"\n--- Agent step {step} ---")
            if step > 1 and self.request_delay:
                time.sleep(self.request_delay)

            remaining = self.max_steps - step + 1
            turn_instructions = (
                f"{SYSTEM_PROMPT}\n\n"
                f"CURRENT TURN: {step} of {self.max_steps}\n"
                f"TURNS REMAINING INCLUDING THIS ONE: {remaining}\n"
            )
            if step >= 5:
                turn_instructions += (
                    "You must make an edit now unless you are genuinely blocked. "
                    "Stop exploring and reserve turns for testing and finalizing.\n"
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
                        tools=TOOLS,
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
                final_message = _item_value(response, "output_text", "") or "Agent finished without a summary."
                return AgentResult(str(final_message), step)

            for call in tool_calls:
                name = _item_value(call, "name")
                raw_arguments = _item_value(call, "arguments", "{}")
                call_id = _item_value(call, "call_id")
                self.log(f"Tool: {name}({raw_arguments})")
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    result = self.workspace.call_tool(name, arguments)
                except Exception as exc:
                    result = f"TOOL ERROR: {type(exc).__name__}: {exc}"
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
                "a change. The harness will run the final tests and prepare the "
                "pull request if they pass."
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


def create_branch(root: Path, issue_number: int) -> str:
    branch = f"agent/issue-{issue_number}"
    existing = _run_process(["git", "branch", "--list", branch], cwd=root).strip()
    if existing:
        raise AgentError(
            f"Branch {branch} already exists locally. Delete or rename it before retrying."
        )
    _run_process(["git", "switch", "--create", branch], cwd=root)
    return branch


def final_test_status(workspace: RepoWorkspace) -> dict[str, Any]:
    raw = workspace.run_tests()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "failed", "output": raw}
    if result.get("status") == "failed":
        raise AgentError(
            "The final test run failed; no commit or pull request was created.\n"
            + str(result.get("output", ""))
        )
    return result


def create_pr(root: Path, issue: dict[str, Any], branch: str, summary: str) -> str:
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
    parser.add_argument("issue_number", type=int, nargs="?", default=None, help="GitHub issue number to fix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the agent and tests, then print the diff without committing, pushing, or opening a PR",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.issue_number is None:
        parser = build_parser()
        parser.error("the following arguments are required: issue_number")
    if args.issue_number <= 0:
        raise AgentError("issue_number must be positive")
    if args.max_steps <= 0:
        raise AgentError("max_steps must be positive")

    root = repository_root(Path.cwd())
    require_clean_worktree(root)
    issue = get_issue(root, args.issue_number)
    print(f"Working on #{issue['number']}: {issue['title']}")

    branch = "(dry run)"
    if not args.dry_run:
        branch = create_branch(root, args.issue_number)
        print(f"Created branch {branch}")

    workspace = RepoWorkspace(root)
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
    pr_url = create_pr(root, issue, branch, result.final_message)
    print(f"Pull request created: {pr_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AgentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent import (
    AgentError,
    AgentRunner,
    GeminiModel,
    RepoWorkspace,
    RunBudget,
    _prepare_mini_sandbox,
    _mini_snapshot,
    _sync_mini_sandbox,
    build_repository_index,
    count_changed_lines,
    find_open_issue_pr,
    final_test_status,
    has_worktree_changes,
    mini_command_allowed,
    mini_edit_required_output,
    mini_test_command,
    mini_command_makes_edit,
    mini_docker_run_args,
    mini_model_kwargs,
    mini_model_name,
)


class FakeResponse:
    def __init__(self, output=None, output_text=""):
        self.output = output or []
        self.output_text = output_text


class ScriptedModel:
    """Exercises the same tool-call loop without calling an API."""

    def __init__(self):
        self.calls = 0

    def create(self, *, instructions, input_items, tools):
        self.calls += 1
        scripts = [
            ("search_code", {"query": "VALUE"}),
            ("read_file", {"path": "app.py", "start_line": 1, "end_line": 10}),
            (
                "edit_file",
                {
                    "path": "app.py",
                    "old_text": "VALUE = 'bug'\n",
                    "new_text": "VALUE = 'fixed'\n",
                },
            ),
            ("run_tests", {}),
        ]
        if self.calls <= len(scripts):
            name, arguments = scripts[self.calls - 1]
            return FakeResponse(
                output=[
                    {
                        "type": "function_call",
                        "name": name,
                        "arguments": json.dumps(arguments),
                        "call_id": f"call-{self.calls}",
                    }
                ]
            )
        return FakeResponse(
            output_text="Changed app.py and confirmed the test suite passes."
        )


class AgentTests(unittest.TestCase):
    def test_run_budget_is_shared_and_bounded(self):
        budget = RunBudget(2)
        budget.consume(phase="localization")
        budget.consume(phase="repair")
        self.assertEqual(budget.remaining, 0)
        self.assertEqual(budget.calls_by_phase, {"localization": 1, "repair": 1})
        with self.assertRaises(AgentError):
            budget.consume(phase="repair")

    def test_agent_runner_stops_when_shared_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "initial",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )

            class ReadOnlyModel:
                def __init__(self):
                    self.calls = 0

                def create(self, *, instructions, input_items, tools, tool_choice=None):
                    self.calls += 1
                    return FakeResponse(
                        output=[
                            {
                                "type": "function_call",
                                "name": "read_file",
                                "arguments": json.dumps(
                                    {
                                        "path": "app.py",
                                        "start_line": 1,
                                        "end_line": 1,
                                    }
                                ),
                                "call_id": f"call-{self.calls}",
                            }
                        ]
                    )

            model = ReadOnlyModel()
            budget = RunBudget(2)
            with self.assertRaises(AgentError):
                AgentRunner(
                    RepoWorkspace(root),
                    model,
                    max_steps=15,
                    budget=budget,
                    log=lambda _: None,
                ).run({"number": 13, "title": "Fix value", "body": "Fix it."})
            self.assertEqual(model.calls, 2)
            self.assertEqual(budget.model_calls_used, 2)

    def test_repository_index_maps_symbols_and_excludes_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text(
                "VALUE = 'bug'\n\n"
                "def fix_value(value):\n"
                "    return value\n",
                encoding="utf-8",
            )
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_app.py").write_text(
                "class AppTest:\n"
                "    def test_value(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            (root / ".env").write_text("SECRET=do-not-index\n", encoding="utf-8")

            index = build_repository_index(root)
            rendered = index.render("Fix the value")

            self.assertIs(build_repository_index(root), index)
            self.assertIn("app.py", rendered)
            self.assertIn("fix_value", rendered)
            self.assertIn("test_value", rendered)
            self.assertIn("VALUE", rendered)
            self.assertNotIn(".env", rendered)
            self.assertNotIn("do-not-index", rendered)

    def test_count_changed_lines_includes_tracked_and_untracked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial"],
                cwd=root, check=True, capture_output=True,
            )
            (root / "tracked.txt").write_text("one\nchanged\n", encoding="utf-8")
            self.assertEqual(count_changed_lines(root), 2)
            self.assertEqual(count_changed_lines(root), 2)  # below a limit of 3
            (root / "new.txt").write_text("new line\nanother\n", encoding="utf-8")
            self.assertEqual(count_changed_lines(root), 4)

    def test_max_diff_lines_cli_option_and_default(self):
        from agent import build_parser
        parser = build_parser()
        self.assertEqual(parser.parse_args(["12"]).max_diff_lines, 200)
        self.assertEqual(parser.parse_args(["12", "--max-diff-lines", "17"]).max_diff_lines, 17)

    def test_verbose_flag(self):
        from agent import build_parser
        parser = build_parser()
        args = parser.parse_args(["12", "--verbose", "--provider", "gemini", "--model", "gemini-model", "--max-steps", "15", "--engine", "mini"])
        self.assertTrue(args.verbose)
        self.assertEqual(args.provider, "gemini")
        self.assertEqual(args.model, "gemini-model")
        self.assertEqual(args.max_steps, 15)
        self.assertEqual(args.engine, "mini")

    def test_mini_engine_uses_explicit_provider_and_isolates_container(self):
        self.assertEqual(
            mini_model_name("gemini", "gemini-3.5-flash-lite"),
            "gemini/gemini-3.5-flash-lite",
        )
        self.assertEqual(mini_model_name("openai", "openai/gpt-5"), "openai/gpt-5")
        self.assertEqual(
            mini_model_kwargs("openai"),
            {"reasoning_effort": "none", "temperature": 0},
        )
        self.assertEqual(mini_model_kwargs("gemini"), {"temperature": 0})
        repository = Path("/tmp/example-repository").resolve()
        docker_args = mini_docker_run_args(repository)
        self.assertIn("none", docker_args)
        self.assertIn("--cap-drop=ALL", docker_args)
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            self.assertIn("--user", docker_args)
            user_index = docker_args.index("--user")
            self.assertEqual(
                docker_args[user_index + 1], f"{os.getuid()}:{os.getgid()}"
            )
        self.assertIn(f"{repository}:/workspace:rw", docker_args)
        self.assertNotIn("GEMINI_API_KEY", docker_args)
        self.assertNotIn("GH_TOKEN", docker_args)
        self.assertFalse(mini_command_makes_edit("rg -n argparse agent.py"))
        self.assertTrue(mini_command_makes_edit("sed -i 's/old/new/' agent.py"))
        self.assertFalse(
            mini_command_makes_edit(
                "python3 -c 'with open(\"agent.py\", \"r\") as f: f.read()'"
            )
        )
        self.assertTrue(
            mini_command_makes_edit(
                "python3 -c 'with open(\"agent.py\", \"a\") as f: f.write(\"x\")'"
            )
        )
        self.assertTrue(
            mini_command_allowed(
                "cat >> agent.py <<'EOF'\nchange\nEOF",
                calls=4,
                edit_made=False,
            )
        )
        self.assertFalse(
            mini_command_allowed(
                "python3 -m unittest tests/test_agent.py",
                calls=4,
                edit_made=False,
            )
        )
        rejected = mini_edit_required_output()
        self.assertEqual(rejected["returncode"], 2)
        self.assertIn("EDIT_REQUIRED", rejected["output"])
        self.assertIn("exception_info", rejected)
        self.assertIn("extra", rejected)
        self.assertIn("unittest", mini_test_command(Path(__file__).parent.parent))
        self.assertTrue(
            mini_command_allowed(
                "python3 -m unittest tests/test_agent.py",
                calls=4,
                edit_made=True,
            )
        )
        self.assertTrue(
            mini_command_makes_edit(
                "python - <<'PY'\nPath('agent.py').write_text('updated')\nPY"
            )
        )

    def test_mini_sandbox_excludes_credentials_and_syncs_safe_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")
            (root / ".env").write_text("TEST_ONLY=placeholder\n", encoding="utf-8")
            sandbox_directory, sandbox_root, before = _prepare_mini_sandbox(root)
            try:
                self.assertFalse((sandbox_root / ".env").exists())
                (sandbox_root / "app.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
                _sync_mini_sandbox(root, sandbox_root, before)
            finally:
                shutil.rmtree(sandbox_directory, ignore_errors=True)
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )
            self.assertEqual(
                (root / ".env").read_text(encoding="utf-8"),
                "TEST_ONLY=placeholder\n",
            )

    def test_mini_python_edit_is_synced_even_when_command_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")
            sandbox_directory, sandbox_root, before = _prepare_mini_sandbox(root)
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; "
                        "Path('app.py').write_text(\"VALUE = 'fixed'\\n\"); "
                        "raise SystemExit(1)",
                    ],
                    cwd=sandbox_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                after = _mini_snapshot(sandbox_root)
                self.assertNotEqual(before, after)
                self.assertNotEqual(result.returncode, 0)
                _sync_mini_sandbox(root, sandbox_root, before)
            finally:
                shutil.rmtree(sandbox_directory, ignore_errors=True)
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )

    def test_worktree_change_detection_includes_untracked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            self.assertFalse(has_worktree_changes(root))
            (root / "app.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
            self.assertTrue(has_worktree_changes(root))

    def test_gemini_thought_signature_is_preserved(self):
        signature = "encrypted-signature"
        raw_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="list_files", arguments="{}"),
            extra_content={"google": {"thought_signature": signature}},
        )
        raw_message = SimpleNamespace(content="", tool_calls=[raw_call])
        raw_response = SimpleNamespace(
            choices=[SimpleNamespace(message=raw_message)]
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return raw_response

        completions = FakeCompletions()
        model = GeminiModel.__new__(GeminiModel)
        model.model = "gemini-3.5-flash-lite"
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        response = model.create(
            instructions="instructions",
            input_items=[{"role": "user", "content": "issue"}],
            tools=[],
        )
        assistant_call = response.output[0]["tool_calls"][0]
        self.assertEqual(
            assistant_call["extra_content"]["google"]["thought_signature"],
            signature,
        )

        next_messages = model._messages(
            "instructions",
            [
                {"role": "user", "content": "issue"},
                *response.output,
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "files",
                },
            ],
        )
        self.assertEqual(
            next_messages[2]["tool_calls"][0]["extra_content"],
            {"google": {"thought_signature": signature}},
        )

    def test_tool_loop_edits_repo_and_runs_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_app.py").write_text(
                "import unittest\n"
                "from app import VALUE\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertEqual(VALUE, 'fixed')\n",
                encoding="utf-8",
            )

            model = ScriptedModel()
            result = AgentRunner(
                RepoWorkspace(root), model, log=lambda _: None
            ).run(
                {
                    "number": 7,
                    "title": "Fix the value",
                    "body": "The value is wrong.",
                }
            )

            self.assertEqual(model.calls, 5)
            self.assertIn("test suite passes", result.final_message)
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )

            test_result = json.loads(RepoWorkspace(root).run_tests())
            self.assertEqual(test_result["status"], "passed")

    def test_tool_budget_gates_search_without_forcing_a_premature_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")

            class WanderingModel:
                def __init__(self):
                    self.calls = 0
                    self.available_tools = []

                def create(
                    self, *, instructions, input_items, tools, tool_choice=None
                ):
                    self.calls += 1
                    self.available_tools.append({tool["name"] for tool in tools})
                    scripts = [
                        ("search_code", {"query": "VALUE"}),
                        ("search_code", {"query": "VALUE"}),
                        ("read_file", {"path": "app.py", "start_line": 1, "end_line": 10}),
                        ("read_file", {"path": "app.py", "start_line": 1, "end_line": 10}),
                        (
                            "edit_file",
                            {
                                "path": "app.py",
                                "old_text": "VALUE = 'bug'\n",
                                "new_text": "VALUE = 'fixed'\n",
                            },
                        ),
                    ]
                    if self.calls <= len(scripts):
                        name, arguments = scripts[self.calls - 1]
                        return FakeResponse(
                            output=[
                                {
                                    "type": "function_call",
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                    "call_id": f"call-{self.calls}",
                                }
                            ]
                        )
                    return FakeResponse(output_text="Fixed the value.")

            model = WanderingModel()
            result = AgentRunner(
                RepoWorkspace(root), model, max_steps=7, log=lambda _: None
                ).run({"number": 9, "title": "Fix value", "body": "Fix it."})

            self.assertEqual(result.steps, 6)
            self.assertNotIn("search_code", model.available_tools[2])
            self.assertEqual(
                model.available_tools[3], {"read_file", "edit_file"}
            )
            self.assertEqual(
                model.available_tools[4], {"edit_file"}
            )
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )

    def test_implementation_phase_requires_edit_after_targeted_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "initial",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )

            class PhaseModel:
                def __init__(self):
                    self.calls = 0
                    self.tools_by_call = []
                    self.choices = []
                    self.input_lengths = []

                def create(
                    self, *, instructions, input_items, tools, tool_choice=None
                ):
                    self.calls += 1
                    self.tools_by_call.append({tool["name"] for tool in tools})
                    self.choices.append(tool_choice)
                    self.input_lengths.append(len(input_items))
                    if self.calls <= 4:
                        name = "read_file"
                        arguments = {
                            "path": "app.py",
                            "start_line": 1,
                            "end_line": self.calls,
                        }
                    elif self.calls == 5:
                        name = "edit_file"
                        arguments = {
                            "path": "app.py",
                            "old_text": "VALUE = 'bug'\n",
                            "new_text": "VALUE = 'fixed'\n",
                        }
                    else:
                        return FakeResponse(output_text="Fixed the value.")
                    return FakeResponse(
                        output=[
                            {
                                "type": "function_call",
                                "name": name,
                                "arguments": json.dumps(arguments),
                                "call_id": f"call-{self.calls}",
                            }
                        ]
                    )

            model = PhaseModel()
            result = AgentRunner(
                RepoWorkspace(root), model, max_steps=6, log=lambda _: None
            ).run({"number": 10, "title": "Fix value", "body": "Fix it."})

            self.assertEqual(result.steps, 6)
            self.assertEqual(
                model.tools_by_call[3], {"search_code", "read_file", "edit_file"}
            )
            self.assertEqual(model.tools_by_call[4], {"edit_file"})
            self.assertIsNone(model.choices[3])
            self.assertEqual(model.input_lengths[4], 1)
            self.assertEqual(
                model.choices[4],
                {"type": "function", "function": {"name": "edit_file"}},
            )
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )

    def test_failed_edit_allows_one_recovery_read_then_requires_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "initial",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )

            class RecoveryModel:
                def __init__(self):
                    self.calls = 0
                    self.tools_by_call = []
                    self.choices = []

                def create(
                    self, *, instructions, input_items, tools, tool_choice=None
                ):
                    self.calls += 1
                    self.tools_by_call.append({tool["name"] for tool in tools})
                    self.choices.append(tool_choice)
                    if self.calls <= 4:
                        name = "read_file"
                        arguments = {
                            "path": "app.py",
                            "start_line": 1,
                            "end_line": self.calls,
                        }
                    elif self.calls == 5:
                        name = "edit_file"
                        arguments = {
                            "path": "app.py",
                            "old_text": "VALUE = 'stale'\n",
                            "new_text": "VALUE = 'fixed'\n",
                        }
                    elif self.calls == 6:
                        name = "read_file"
                        arguments = {
                            "path": "app.py",
                            "start_line": 1,
                            "end_line": 1,
                        }
                    elif self.calls == 7:
                        name = "edit_file"
                        arguments = {
                            "path": "app.py",
                            "old_text": "VALUE = 'bug'\n",
                            "new_text": "VALUE = 'fixed'\n",
                        }
                    else:
                        return FakeResponse(output_text="Fixed the value.")
                    return FakeResponse(
                        output=[
                            {
                                "type": "function_call",
                                "name": name,
                                "arguments": json.dumps(arguments),
                                "call_id": f"call-{self.calls}",
                            }
                        ]
                    )

            model = RecoveryModel()
            result = AgentRunner(
                RepoWorkspace(root), model, max_steps=8, log=lambda _: None
            ).run({"number": 11, "title": "Fix value", "body": "Fix it."})

            self.assertEqual(result.steps, 8)
            self.assertEqual(model.tools_by_call[4], {"edit_file"})
            self.assertEqual(model.tools_by_call[5], {"read_file", "edit_file"})
            self.assertEqual(model.tools_by_call[6], {"edit_file"})
            self.assertIsNone(model.choices[5])
            self.assertEqual(
                model.choices[6],
                {"type": "function", "function": {"name": "edit_file"}},
            )
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )

    def test_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = RepoWorkspace(Path(directory))
            with self.assertRaises(ValueError):
                workspace.read_file("../outside.txt")
            with self.assertRaises(ValueError):
                workspace.edit_file(".git/config", "", "bad")
            with self.assertRaises(ValueError):
                workspace.edit_file(".github/workflows/agent.yml", "", "bad")
            with self.assertRaises(ValueError):
                workspace.edit_file(".env", "", "SECRET=bad")

    def test_read_file_can_target_a_line_range(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = RepoWorkspace(Path(directory))
            Path(directory, "sample.py").write_text(
                "one\ntwo\nthree\nfour\n", encoding="utf-8"
            )
            self.assertEqual(
                workspace.read_file("sample.py", start_line=2, end_line=3),
                "2: two\n3: three",
            )

    def test_edit_file_requires_one_exact_match(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = RepoWorkspace(Path(directory))
            workspace.edit_file("new.txt", "", "hello")
            self.assertEqual(workspace.read_file("new.txt"), "1: hello")
            self.assertIn(
                "not found",
                workspace.call_tool("edit_file", {"path": "new.txt", "old_text": "x", "new_text": "y"}),
            )

    def test_missing_test_configuration_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            result = json.loads(RepoWorkspace(Path(directory)).run_tests())
            self.assertEqual(result["status"], "skipped")

    def test_skipped_final_tests_block_pull_request(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AgentError):
                final_test_status(RepoWorkspace(Path(directory)))

    @patch("agent._run_process")
    def test_existing_issue_pull_request_is_detected(self, run_process):
        run_process.return_value = json.dumps(
            [
                {
                    "number": 12,
                    "title": "Fix #7: Example",
                    "body": "Automated fix\n\nFixes #7",
                    "url": "https://github.com/example/repo/pull/12",
                }
            ]
        )
        self.assertEqual(
            find_open_issue_pr(Path("."), 7),
            "https://github.com/example/repo/pull/12",
        )

    def test_run_tests_does_not_inherit_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "check_environment.py"
            script.write_text(
                "import os\n"
                "names = ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN')\n"
                "assert not any(os.environ.get(name) for name in names)\n",
                encoding="utf-8",
            )
            environment = {
                "TEST_COMMAND": f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
                "GEMINI_API_KEY": "test-gemini-secret",
                "OPENAI_API_KEY": "test-openai-secret",
                "GH_TOKEN": "test-github-token",
                "GITHUB_TOKEN": "test-github-token",
            }
            with patch.dict(os.environ, environment, clear=False):
                result = json.loads(RepoWorkspace(root).run_tests())

            self.assertEqual(result["status"], "passed")

    def test_changed_worktree_survives_turn_limit_for_final_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "app.py").write_text("VALUE = 'bug'\n", encoding="utf-8")

            class EditOnlyModel:
                def create(self, *, instructions, input_items, tools):
                    return FakeResponse(
                        output=[
                            {
                                "type": "function_call",
                                "name": "edit_file",
                                "arguments": json.dumps(
                                    {
                                        "path": "app.py",
                                        "old_text": "VALUE = 'bug'\n",
                                        "new_text": "VALUE = 'fixed'\n",
                                    }
                                ),
                                "call_id": "edit-1",
                            }
                        ]
                    )

            result = AgentRunner(
                RepoWorkspace(root), EditOnlyModel(), max_steps=1, log=lambda _: None
            ).run({"number": 8, "title": "Fix value", "body": "Fix it."})

            self.assertEqual(result.steps, 1)
            self.assertIn("final tests", result.final_message)
            self.assertIn("app.py", result.final_message)
            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"),
                "VALUE = 'fixed'\n",
            )


if __name__ == "__main__":
    unittest.main()

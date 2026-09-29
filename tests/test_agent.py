import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent import AgentRunner, GeminiModel, RepoWorkspace


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
            ("list_files", {}),
            ("read_file", {"path": "app.py"}),
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

    def test_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = RepoWorkspace(Path(directory))
            with self.assertRaises(ValueError):
                workspace.read_file("../outside.txt")
            with self.assertRaises(ValueError):
                workspace.edit_file(".git/config", "", "bad")

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


if __name__ == "__main__":
    unittest.main()

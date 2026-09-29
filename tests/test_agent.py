import json
import tempfile
import unittest
from pathlib import Path

from agent import AgentRunner, RepoWorkspace


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
            ("write_file", {"path": "app.py", "content": "VALUE = 'fixed'\n"}),
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
                workspace.write_file(".git/config", "bad")

    def test_missing_test_configuration_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            result = json.loads(RepoWorkspace(Path(directory)).run_tests())
            self.assertEqual(result["status"], "skipped")


if __name__ == "__main__":
    unittest.main()

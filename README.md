# GitHub issue agent MVP

This repository contains a small issue-to-pull-request agent. It fetches one GitHub issue with `gh`, gives the issue to Gemini or OpenAI, and lets the model use four repository tools:

- `read_file`
- `search_code`
- `edit_file`
- `run_tests`

The repository file list is supplied in the initial issue context, so the model does not spend a turn discovering it. The model cannot run `git`, create commits, push, or open pull requests. The Python harness performs those actions only after the agent stops, the worktree has a diff, and a final test run passes. Gemini uses Google's OpenAI-compatible Chat Completions endpoint for the tool loop. Existing files are changed with exact small replacements through `edit_file`, which keeps the agent's requests smaller.

The agent has twelve paced model turns and receives its current turn and remaining budget on every request. If it uses its final turn to make a change, the harness still runs the final tests and prepares the pull request when they pass.

`search_code` returns line numbers with nearby context, and `read_file` requires a focused line range capped at 200 lines. This lets the agent jump to the relevant part of a large file instead of repeatedly sending its entire contents to the model.

## Install

Run this from a clone of the repository you want the agent to modify:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
gh auth login
export GEMINI_API_KEY="your-gemini-key"
```

Gemini is the default provider and `gemini-3.5-flash-lite` is the default model. The agent uses at most twelve model turns and spaces Gemini requests by five seconds by default. Set `GEMINI_MODEL` if needed. To use OpenAI instead:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-openai-key"
export OPENAI_MODEL="gpt-5.6"
```

The repository must have a clean worktree before every run. Use a disposable clone when testing because the agent is allowed to edit files.

## Test the implementation without GitHub or an API key

The unit tests use a scripted fake model, so they exercise path safety, tool dispatch, editing, and test execution without making network calls:

```bash
python -m unittest discover -s tests -v
```

## Run locally against a real issue

Create a small issue in the target repository, for example one describing a failing test or a contained bug. Then run a dry run from a clean disposable clone:

```bash
git status --short
python agent.py 12 --dry-run
```

Dry-run still fetches the issue and calls the model, and it can edit the checkout. It does not create a branch, commit, push, or pull request. Review the printed diff, then restore the disposable clone before trying again.

For the full issue-to-PR flow:

```bash
python agent.py 12
```

This creates `agent/issue-12`, runs the agent, runs the detected test command one final time, commits the changes, pushes the branch, and opens a PR with `Fixes #12`. If the final tests fail or the agent makes no changes, it stops before creating the PR.

The test command is detected in this order: `TEST_COMMAND`, `npm test`, `cargo test`, `go test ./...`, pytest configuration, and Python `unittest` tests under `tests/`. For an existing project, set the exact command explicitly when detection is not appropriate:

```bash
export TEST_COMMAND='python -m pytest -q'
```

## Run it from GitHub Actions

The included workflow supports two triggers:

1. Run **Actions → Agent fix → Run workflow** and enter an issue number.
2. Open a new issue. The agent starts automatically and uses that issue number.

Before enabling the workflow, add an Actions repository secret named `GEMINI_API_KEY`. Optionally add `GEMINI_MODEL` as a repository variable. Set `LLM_PROVIDER` to `openai` and add `OPENAI_API_KEY` instead if you want to use OpenAI. The workflow grants the job permission to push branches and create pull requests, and uses the built-in `GH_TOKEN` for GitHub CLI authentication.

Every newly opened issue starts a run, so configure repository access and issue permissions accordingly. Review the generated PR and keep normal branch protection and CI checks enabled.

# GitHub issue agent MVP

[![CI](https://github.com/gdawerty/Agent_Test/actions/workflows/ci.yml/badge.svg)](https://github.com/gdawerty/Agent_Test/actions/workflows/ci.yml)

This repository contains a small issue-to-pull-request agent. It fetches one GitHub issue with `gh`, gives the issue to Gemini or OpenAI, and lets the model use four repository tools:

- `read_file`
- `search_code`
- `edit_file`
- `run_tests`

The initial issue context includes a deterministic Python repository map with files, classes, functions, methods, tests, and line ranges. The model cannot run `git`, create commits, push, or open pull requests. The Python harness performs those actions only after the agent stops, the worktree has a diff, and a final test run passes. Gemini uses Google's OpenAI-compatible Chat Completions endpoint for the tool loop. Existing files are changed with exact small replacements through `edit_file`, which keeps the agent's requests smaller.

The agent has twelve paced model calls by default and receives its current phase and remaining budget on every request. Investigation, implementation, repair, and verification have separate budgets, so exploration cannot consume the calls reserved for editing. If the mini engine is selected and does not produce a change, the focused engine receives only the unused calls from that same global budget. If it uses its final call to make a change, the harness still runs the final tests and prepares the pull request when they pass.

The default custom phase budgets are 4 investigation calls, 3 implementation calls, 3 repair calls, and 2 verification/finalization calls. Repeated identical reads or searches are reported as already observed. The mini engine is limited to 6 calls before the remaining global budget can be offered to the focused engine.

The harness also limits exploration: the model gets at most two code searches, loses search access after editing, and can use targeted reads during the investigation phase. Starting at the implementation checkpoint, the provider is explicitly asked for `edit_file` and receives no read or search tool. If an exact replacement fails, one recovery turn can read the affected code before the next edit attempt. A shared run budget covers every engine, including mini-to-custom fallback, so fallback cannot silently start a second full budget.

`search_code` returns line numbers with nearby context, and `read_file` requires a focused line range capped at 200 lines. The AST map lets the agent jump directly to likely symbols and relevant tests instead of spending turns discovering repository structure. Indexes are cached by checkout revision for the lifetime of the process.

## Install

Run this from a clone of the repository you want the agent to modify:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
gh auth login
export GEMINI_API_KEY="your-gemini-key"
```

Gemini is the default provider and `gemini-3.5-flash-lite` is the default model. The agent uses at most twelve model calls and spaces Gemini requests by five seconds by default. Set `GEMINI_MODEL` if needed. To use OpenAI instead:

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
2. Create an issue, review it, and add the `agent-approved` label. The agent then starts automatically for that issue.

Before enabling the workflow, add an Actions repository secret named `GEMINI_API_KEY`. Optionally add `GEMINI_MODEL` as a repository variable. Set `LLM_PROVIDER` to `openai` and add `OPENAI_API_KEY` instead if you want to use OpenAI. The workflow grants the job permission to push branches and create pull requests, and uses the built-in `GH_TOKEN` for GitHub CLI authentication.

Only issues labeled `agent-approved` start an automatic run. The job also uses the `agent-fix` environment. Configure that environment in **Settings → Environments** with yourself or another trusted maintainer as a required reviewer; this adds a second approval gate before the runner receives the workflow credentials.

Protect the `main` branch in **Settings → Rules → Rulesets** or **Settings → Branches**. Require pull requests, at least one approving review, the `CI / test` status check, and no direct pushes or force pushes. Keep automatic merging disabled for agent-created pull requests until a maintainer reviews the diff.

The harness refuses agent edits to workflow, action, credential, and private-key paths. It runs tests with a minimal environment and refuses to create a pull request when tests are skipped. If an issue already has an open automated pull request, a rerun reports that pull request instead of creating a duplicate; a stale branch receives a retry suffix.

## Optional mini-SWE-agent engine

The default engine remains the small custom tool loop above. This repository also includes an experimental mini-SWE-agent engine for comparison. It uses mini-SWE-agent's single bash tool, LiteLLM's Gemini adapter, and a Docker execution environment, while this outer harness continues to own branch creation, protected-path validation, final tests, commits, pushes, and pull requests.

The optional dependency is deliberately separate from `requirements.txt`:

```bash
python -m pip install -r requirements-mini.txt
docker build -f Dockerfile.agent -t agent-fix-sandbox:latest .
```

Run it locally from a clean checkout with a real issue number:

```bash
export GEMINI_API_KEY="your-gemini-key"
python agent.py 12 --engine mini --max-steps 12 --dry-run
```

The Docker container has no network, receives no `GH_TOKEN`, `GITHUB_TOKEN`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`, and mounts a temporary copy at `/workspace`. The copy excludes `.git`, ignored virtualenv/build directories, and credential-looking files; safe changes are synchronized back after the run. The outer process still rejects protected-file changes and requires a passing final test run before creating a PR. This is an experimental containment boundary, not a complete security sandbox.

To try it through GitHub Actions after the local dry run works, add a repository variable named `AGENT_ENGINE` with the value `mini`. The existing `agent-approved` label and environment reviewer gates still apply. Delete the variable or set it to `custom` to return to the current engine.

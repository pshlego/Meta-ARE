# GitHub Copilot instructions (repo-wide)

## Environment & command execution (IMPORTANT)
- This repo uses a Python environment managed by `uv`.
- **Before running any Python-related command** (anything that invokes `python`, `pip`, `pytest`, `ruff`, `mypy`, `alembic`, `django-admin`, `flask`, `uvicorn`, `gunicorn`, `mkdocs`, etc.), always run it **inside the uv environment**.
- Prefer these patterns (pick the most direct one):
  - `uv run <command ...>` (preferred for one-off commands)
  - If an interactive shell is needed: `uv venv && source .venv/bin/activate` (or platform equivalent), then run commands.
- Do **not** assume the env is already active.
- Only skip `uv` wrapping for common non-Python shell commands like: `ls`, `cd`, `cat`, `echo`, `pwd`, `git ...`, `mkdir`, `rm`, `cp`, `mv`, `sed`, `awk`, `curl`, `wget` (unless they are part of a Python workflow step).

## Examples
- ✅ `uv run python -m pytest -q`
- ✅ `uv run ruff check .`
- ✅ `uv run python -m mypy src/`
- ✅ `uv run python scripts/sync_data.py --dry-run`
- ✅ `git status` (no `uv` needed)
- ✅ `ls -la` (no `uv` needed)
- ❌ `pytest` (should be `uv run pytest`)
- ❌ `python script.py` (should be `uv run python script.py`)

## Dependency management
- Add/remove deps using `uv add <pkg>` / `uv remove <pkg>` (or whatever standard this repo uses).
- If a lockfile is present, keep it updated consistently with `uv` workflows.

## Documentation & consistency (IMPORTANT)
- **After any code changes**, always check and update relevant documentation (only the docs in `are/simulation/checkpoint`, do not touch other docs that come with the original repo), output-related files (docstring, config references), and log/result schemas to reflect the latest changes.
- If inconsistencies are found between code and documentation, always correct them. If unsure how to resolve an inconsistency, ask the user before proceeding.
- Keep CLI help text, config file comments, and doc examples in sync with actual implementations.
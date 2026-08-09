# Project: IDD Climate Models

## Session Memory Instructions
**IMPORTANT:** Update `.claude/memory.md` after EVERY substantive action (running commands, completing task steps, changing focus). This ensures context is preserved if the session dies unexpectedly.

The memory file should always contain:
1. Current task summary
2. Why we're doing it
3. Next steps
4. A ready-to-paste resume prompt

## Project Overview
Climate modeling pipeline for infectious disease dynamics (IDD). Key components:
- `src/idd_climate_models/01_process_raw_through_climada_input/` - Process raw climate data for TC input
- `src/idd_climate_models/01_run_tc_risk/` - Run tropical cyclone risk models
- `src/idd_climate_models/02_run_climada/` - Run CLIMADA impact modeling

## Environment scheme: venv-only (declared 2026-08-09)

This repo runs the **pure-uv scheme**: a project `.venv` at the repo root,
uv-managed interpreter, uv-managed dependencies. One scheme per repo, never
mixed.

- **Expected:** `./.venv` exists and is the only environment for this repo.
  Its interpreter is a uv-managed standalone CPython (`uv python install`),
  never a conda python. All code runs via `.venv/bin/python` — by absolute
  path in Slurm/jobmon task commands, no activation required.
- **Forbidden:** conda hosting of any kind for this repo — no conda env named
  after the repo, no `environment.yml`, no `UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX`
  idiom. Those belong to the retired conda scheme.
- **Session start:** do NOT activate a conda env (this overrides the global
  "activate the matching conda environment" rule — there is no matching env).
  Use `.venv/bin/python` directly.
- **Sync idiom:** `uv sync --inexact` day-to-day; plain `uv sync` (exact) for
  clean rebuilds, vetted first with `uv sync --dry-run`. `--all-extras` to get
  the full surface including the `notebooks`/`climada`/`downloads` extras.
- **jobmon:** core dependency (`jobmon_installer_ihme==10.12.2`, IHME
  artifactory index in `pyproject.toml`); orchestrators run from the .venv and
  worker command templates invoke `.venv/bin/python` by absolute path.
- Interpreter downloads are deliberate only: the user-level uv policy is
  `python-downloads = "manual"` — a plain `uv venv`/`uv sync` must never
  trigger a CPython download.
- Uses IHME's jobmon for workflow orchestration.

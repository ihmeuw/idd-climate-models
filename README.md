# IDD Climate Models

Pipeline for estimating tropical cyclone (TC) mortality risk under climate change scenarios.

## Two Tracks

### 1. Direct Relative Risk Modeling (Active)
Parametric and ML models predicting TC mortality from wind speed and sociodemographic index (SDI).

**Status:** Stage-based model comparison complete. Double hurdle (DH) is the best structure (mae_rate 1.28 vs 1.49 for single). Final model selection in progress.

**Entry point:** `notebooks/dh_model_analysis.ipynb`

**Code:** `src/idd_climate_models/direct_relative_risk/` and `src/idd_climate_models/tc_models/`

### 2. TC Risk Pipeline (Paused)
Raw climate data → TC risk models → CLIMADA impact assessment.

**Code:** `src/idd_climate_models/01_process_raw_through_climada_input/`, `src/idd_climate_models/01_run_tc_risk/`, `src/idd_climate_models/02_run_climada/`

---

## Environment

This repo uses the fleet-standard pure-uv scheme: a project `.venv` at the
repo root on a uv-managed CPython 3.12, with all dependencies declared in
`pyproject.toml` / `uv.lock`. No conda.

```bash
cd <repo>
uv venv --python 3.12        # .venv on the managed CPython
uv sync --all-extras         # exact sync from uv.lock; dev group included
.venv/bin/python -m ipykernel install --user --name idd-climate-models \
    --display-name "idd-climate-models (.venv)"
```

- Run code via `.venv/bin/python` — by absolute path in Slurm/jobmon task
  commands (no activation required); `source .venv/bin/activate` is an
  optional interactive convenience.
- Day-to-day dependency updates: `uv sync --inexact`. Clean rebuilds: plain
  `uv sync` (vet first with `uv sync --dry-run`). Never `pip install` into
  the .venv — declare the dep in `pyproject.toml` instead.

---

## Direct Risk Modeling — Quick Start

```python
from idd_climate_models.tc_models import load_tc_data

tc_df = load_tc_data()
```

See `src/idd_climate_models/direct_relative_risk/README.md` for full API docs, and `model_universe.md` in that directory for the model architecture overview.

---

## Key Results (as of 2026-03-23)

| Structure | mae_rate |
|-----------|----------|
| Double Hurdle (DH) | **1.28** |
| POT | 1.38 |
| Hurdle | 1.44 |
| Single | 1.49 |

Best DH configuration: threshold=90, RF s1, logistic s2, XGBoost bulk, GPD tail.

---

## Contact

Bobby Creiner — bcreiner@uw.edu

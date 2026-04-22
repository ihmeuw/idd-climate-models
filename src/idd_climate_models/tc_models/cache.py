"""
cache.py
Model caching and manifest tracking.

Each spec's results are stored in a per-spec JSON file at RESULTS_DIR/{spec_id}.json.
This avoids concurrent write conflicts when thousands of jobs run in parallel.
load_manifest() aggregates all per-spec files into a single DataFrame.

Insample fits and per-fold OOS fits are saved as pkl files.

Functions
---------
  spec_to_id(spec)                               → 12-char hex str
  model_path(spec_id, fit_type, seed, fold)      → Path
  model_exists(spec_id, fit_type, seed, fold)    → bool
  save_model(spec_id, fit_type, fitted, seed, fold) → Path
  load_model(spec_id, fit_type, seed, fold)      → fitted | None

  log_to_manifest(spec_id, spec, fit_type, metrics, seed, fold)
  is_logged(spec_id, fit_type, seed, fold)       → bool
  load_manifest()                                → pd.DataFrame
"""

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .constants import MODELS_DIR, RESULTS_DIR, STAGE_MODELS_DIR, STAGE_RESULTS_DIR

# Fields excluded from the ID hash
_EXCLUDE_FROM_ID = {'label'}

_SPEC_FIELDS = [
    'structure', 'dist',
    's1_dist', 's2_dist', 'bulk_dist', 'tail_dist',
    'threshold_pct', 'covars',
    's1_covars', 's2_covars', 'bulk_covars', 'tail_covars',
    'interaction', 'island_interact',
]

_METRIC_COLS = [
    'mae_rate', 'rmse_rate', 'mae_log', 'cor_rate',
    'mae_count', 'rmse_count', 'cor_count', 'zero_acc',
    'skill_mae_rate', 'skill_rmse_rate', 'skill_mae_count', 'skill_rmse_count',
    'sd_mae_rate', 'sd_mae_count', 'n_tests',
]


def spec_to_id(spec: Dict) -> str:
    """Deterministic 12-char hex ID for a spec dict. 'label' is excluded."""
    clean = {k: v for k, v in spec.items() if k not in _EXCLUDE_FROM_ID}
    canonical = json.dumps(clean, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def _spec_result_path(spec_id: str) -> Path:
    """Per-spec JSON result file. One file per spec, written by one worker."""
    return RESULTS_DIR / f'{spec_id}.json'


# ── Model file I/O ────────────────────────────────────────────────────────────

def model_path(spec_id: str, fit_type: str,
               seed: Optional[int] = None, fold: Optional[int] = None) -> Path:
    if fit_type == 'insample':
        return MODELS_DIR / f'{spec_id}_insample.pkl'
    elif fit_type == 'oos':
        if seed is None or fold is None:
            raise ValueError("seed and fold required for fit_type='oos'")
        return MODELS_DIR / f'{spec_id}_seed{seed}_fold{fold}.pkl'
    else:
        raise ValueError(f"Unknown fit_type '{fit_type}'. Expected 'insample' or 'oos'.")


def model_exists(spec_id: str, fit_type: str,
                 seed: Optional[int] = None, fold: Optional[int] = None) -> bool:
    return model_path(spec_id, fit_type, seed, fold).exists()


def save_model(spec_id: str, fit_type: str, fitted: Any,
               seed: Optional[int] = None, fold: Optional[int] = None) -> Path:
    path = model_path(spec_id, fit_type, seed, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(spec_id: str, fit_type: str,
               seed: Optional[int] = None, fold: Optional[int] = None) -> Optional[Any]:
    path = model_path(spec_id, fit_type, seed, fold)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Result logging (per-spec JSON) ───────────────────────────────────────────

def log_to_manifest(
    spec_id:  str,
    spec:     Dict,
    fit_type: str,
    metrics:  Dict,
    seed:     Optional[int] = None,
    fold:     Optional[int] = None,
) -> None:
    """
    Append one result entry to the per-spec JSON file at RESULTS_DIR/{spec_id}.json.
    Each spec has its own file — no concurrent write conflicts between parallel jobs.
    """
    path = _spec_result_path(spec_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry: Dict[str, Any] = {
        'spec_id':   spec_id,
        'fit_type':  fit_type,
        'seed':      seed,
        'fold':      fold,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    for field in _SPEC_FIELDS:
        entry[field] = spec.get(field)
    for col in _METRIC_COLS:
        entry[col] = metrics.get(col)

    # Load existing entries, append, write back
    if path.exists():
        try:
            with open(path) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt file — start fresh
            entries = []
    else:
        entries = []

    entries.append(entry)
    with open(path, 'w') as f:
        json.dump(entries, f)


def is_logged(spec_id: str, fit_type: str,
              seed: Optional[int] = None, fold: Optional[int] = None) -> bool:
    """Check whether a (spec_id, fit_type, seed, fold) entry exists."""
    path = _spec_result_path(spec_id)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable file — treat as not logged
        return False
    for e in entries:
        if e.get('fit_type') != fit_type:
            continue
        if seed is None and e.get('seed') is not None:
            continue
        if seed is not None and e.get('seed') != seed:
            continue
        if fold is None and e.get('fold') is not None:
            continue
        if fold is not None and e.get('fold') != fold:
            continue
        return True
    return False


# ============================================================================
# Stage cache (new architecture)
# ============================================================================

_STAGE_FIELDS = ['stage_type', 'dist', 'covars', 'threshold_pct']

_STAGE_METRIC_COLS = [
    # count metrics
    'mae_rate', 'rmse_rate', 'mae_log', 'cor_rate',
    'mae_count', 'rmse_count', 'cor_count', 'zero_acc',
    # binary metrics
    'brier_score', 'log_loss', 'roc_auc',
    # supplemental
    'n_obs', 'n_tail_train', 'threshold',
]


def stage_to_id(stage: dict) -> str:
    """Deterministic 12-char hex ID for a stage dict."""
    canonical = json.dumps(stage, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def stage_model_path(stage_id: str, fit_type: str,
                     seed: Optional[int] = None,
                     fold: Optional[int] = None) -> Path:
    if fit_type == 'insample':
        return STAGE_MODELS_DIR / f'{stage_id}_insample.pkl'
    elif fit_type == 'oos':
        if seed is None or fold is None:
            raise ValueError("seed and fold required for fit_type='oos'")
        return STAGE_MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'
    else:
        raise ValueError(f"Unknown fit_type '{fit_type}'")


def stage_model_exists(stage_id: str, fit_type: str,
                       seed: Optional[int] = None,
                       fold: Optional[int] = None) -> bool:
    return stage_model_path(stage_id, fit_type, seed, fold).exists()


def save_stage_model(stage_id: str, fit_type: str, fitted: Any,
                     seed: Optional[int] = None,
                     fold: Optional[int] = None) -> Path:
    path = stage_model_path(stage_id, fit_type, seed, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_stage_model(stage_id: str, fit_type: str,
                     seed: Optional[int] = None,
                     fold: Optional[int] = None) -> Optional[Any]:
    path = stage_model_path(stage_id, fit_type, seed, fold)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _stage_result_path(stage_id: str) -> Path:
    return STAGE_RESULTS_DIR / f'{stage_id}.json'


def log_stage_result(
    stage_id:  str,
    stage:     dict,
    fit_type:  str,
    metrics:   dict,
    seed:      Optional[int] = None,
    fold:      Optional[int] = None,
) -> None:
    """Append one result entry to the per-stage JSON at STAGE_RESULTS_DIR/{stage_id}.json."""
    path = _stage_result_path(stage_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry: Dict[str, Any] = {
        'stage_id': stage_id,
        'fit_type': fit_type,
        'seed':     seed,
        'fold':     fold,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    for field in _STAGE_FIELDS:
        entry[field] = stage.get(field)
    for col in _STAGE_METRIC_COLS:
        entry[col] = metrics.get(col)

    if path.exists():
        try:
            with open(path) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            entries = []
    else:
        entries = []

    entries.append(entry)
    with open(path, 'w') as f:
        json.dump(entries, f)


def is_stage_logged(stage_id: str, fit_type: str,
                    seed: Optional[int] = None,
                    fold: Optional[int] = None) -> bool:
    path = _stage_result_path(stage_id)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    for e in entries:
        if e.get('fit_type') != fit_type:
            continue
        if seed is None and e.get('seed') is not None:
            continue
        if seed is not None and e.get('seed') != seed:
            continue
        if fold is None and e.get('fold') is not None:
            continue
        if fold is not None and e.get('fold') != fold:
            continue
        return True
    return False


def load_stage_manifest() -> 'pd.DataFrame':
    """Aggregate all per-stage JSON files into a single DataFrame."""
    if not STAGE_RESULTS_DIR.exists():
        return pd.DataFrame()
    result_files = list(STAGE_RESULTS_DIR.glob('*.json'))
    if not result_files:
        return pd.DataFrame()
    all_entries: List[Dict] = []
    for f in result_files:
        try:
            with open(f) as fh:
                all_entries.extend(json.load(fh))
        except Exception:
            pass
    return pd.DataFrame(all_entries) if all_entries else pd.DataFrame()


def load_manifest() -> pd.DataFrame:
    """Aggregate all per-spec JSON files into a single DataFrame."""
    if not RESULTS_DIR.exists():
        return pd.DataFrame()
    result_files = list(RESULTS_DIR.glob('*.json'))
    if not result_files:
        return pd.DataFrame()
    all_entries: List[Dict] = []
    for f in result_files:
        try:
            with open(f) as fh:
                all_entries.extend(json.load(fh))
        except Exception:
            pass
    return pd.DataFrame(all_entries) if all_entries else pd.DataFrame()

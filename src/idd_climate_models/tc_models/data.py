"""
data.py
Data loading and preparation for TC mortality modeling.

Two entry points
----------------
load_tc_data(path=None)
    Load from disk (default path from constants.py) and prepare.
    This is what notebooks and scripts should call.

prepare_tc_data(df)
    Apply all cleaning and feature engineering to an already-loaded
    raw DataFrame.  Useful for testing, or when loading from a
    non-standard source (e.g. a database query, a different vintage).

Columns added by prepare_tc_data
---------------------------------
    death_y_n          : int   1 if total_deaths > 0, else 0
    log_exp            : float log(exposed_population)
    death_rate         : float total_deaths / exposed_population
    log_sdi            : float log(sdi)
    log_max_wind_speed : float log(max_wind_speed)
    neg_sdi            : float -sdi  (used in some model specs)
    data_year          : int   copy of 'year' (engine expects this name)

Usage
-----
    from idd_climate_models.tc_models.data import load_tc_data
    tc_df = load_tc_data()

    # Override path (e.g. updated vintage):
    tc_df = load_tc_data(path='/path/to/new_file.csv')

    # Already have a DataFrame from another source:
    from idd_climate_models.tc_models.data import prepare_tc_data
    tc_df = prepare_tc_data(raw_df)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import (
    TC_DATA_PATH,
    COLUMNS_TO_DROP,
    COLUMN_RENAMES,
    INTEGER_COLS,
    MIN_EXPOSED_POPULATION,
    MAX_DATA_YEAR,
    LOCATION_LEVEL,
)


# ============================================================================
# Public: prepare_tc_data
# ============================================================================

def prepare_tc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and feature-engineer a raw TC DataFrame.

    Parameters
    ----------
    df : raw DataFrame as loaded from the ibtracs CSV

    Returns
    -------
    Filtered, typed, feature-engineered DataFrame ready for
    run_model_evaluation().  Original DataFrame is not modified.
    """
    df = df.copy()

    # ── Drop unused columns ───────────────────────────────────────────────────
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # ── Rename date duplicates ────────────────────────────────────────────────
    df = df.rename(columns=COLUMN_RENAMES)

    # ── Type coercion ─────────────────────────────────────────────────────────
    df['sdi']   = pd.to_numeric(df['sdi'],   errors='coerce')

    for col in INTEGER_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)


    # ── Row filters ───────────────────────────────────────────────────────────
    if MIN_EXPOSED_POPULATION == 0:
        exposed_filter: pd.Series = df['exposed_population'] > MIN_EXPOSED_POPULATION
    else:
        exposed_filter: pd.Series = df['exposed_population'] >= MIN_EXPOSED_POPULATION

    mask: pd.Series = (
        exposed_filter
        & (df['year'] <= MAX_DATA_YEAR)
        & (df['level'] == LOCATION_LEVEL)
    )
    df = df.loc[mask].copy()  # type: ignore[index]

    # ── Derived columns required by the modeling engine ───────────────────────
    df['death_y_n']          = (df['total_deaths'] > 0).astype(int)
    df['log_exp']            = np.log(df['exposed_population'])
    df['death_rate']         = df['total_deaths'] / df['exposed_population']
    df['log_sdi']            = np.log(df['sdi'])
    df['log_max_wind_speed'] = np.log(df['max_wind_speed'])
    df['neg_sdi']            = -df['sdi']

    # ── Year column normalisation ─────────────────────────────────────────────
    # engine.py expects 'data_year'; raw file has 'year'
    if 'year' in df.columns and 'data_year' not in df.columns:
        df['data_year'] = df['year']

    return df.reset_index(drop=True)


# ============================================================================
# Public: load_tc_data
# ============================================================================

def load_tc_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the TC mortality dataset from disk and prepare it.

    Parameters
    ----------
    path : optional Path or str override for TC_DATA_PATH in constants.py.
           Pass this when using a different data vintage or running tests.

    Returns
    -------
    Prepared DataFrame ready for run_model_evaluation().

    Raises
    ------
    FileNotFoundError if the resolved path does not exist.
    ValueError if the 'NA' (North Atlantic) basin is missing after load.
    """
    resolved = Path(path) if path is not None else TC_DATA_PATH

    if not resolved.exists():
        raise FileNotFoundError(
            f"TC data file not found: {resolved}\n"
            f"Update TC_DATA_PATH in constants.py or pass a path= argument."
        )

    df = pd.read_csv(
        resolved,
        keep_default_na=False,   # CRITICAL: prevents 'NA' (North Atlantic) becoming NaN
        na_values=[''],           # only blank cells become NaN
        dtype={'basin': str},     # CRITICAL: force basin to str before any NA inference
    )
    df['basin'] = df['basin'].fillna('').astype(str)
    df['basin'] = df['basin'].replace('', 'NA')

    return prepare_tc_data(df)
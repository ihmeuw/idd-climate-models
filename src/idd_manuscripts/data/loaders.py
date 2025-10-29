"""Data loading utilities."""

import pandas as pd
from pathlib import Path


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file with basic error handling."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading {filepath}: {str(e)}")

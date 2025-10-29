"""Statistical analysis functions."""

import pandas as pd
import numpy as np
from typing import Tuple


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate basic summary statistics."""
    return df.describe()


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()

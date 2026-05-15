"""
utils/detect_problem.py
Automatically detect classification vs regression from the target column.
"""

import pandas as pd


def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()

    # Non-numeric → classification
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    n_unique = series.nunique()
    n_total  = len(series)

    # Few unique integers → classification (e.g. 0/1, 1/2/3)
    if n_unique <= 15:
        return "classification"

    # Low unique ratio → classification
    if n_unique / n_total < 0.05:
        return "classification"

    return "regression"

"""
utils/feature_engineering.py
Automated feature engineering using Featuretools Deep Feature Synthesis.
Works on the already-preprocessed numeric DataFrame (X).
"""

import pandas as pd
from config import FEATURE_ENGINEERING_DEPTH


def run_feature_engineering(X: pd.DataFrame) -> tuple:
    """
    Run Featuretools DFS on X.
    Returns (X_new, report_message).
    Falls back gracefully if featuretools fails.
    """
    try:
        import featuretools as ft

        df = X.copy().reset_index(drop=True)
        df["_index"] = df.index

        es = ft.EntitySet(id="data")
        es = es.add_dataframe(
            dataframe_name="dataset",
            dataframe=df,
            index="_index"
        )

        feature_matrix, _ = ft.dfs(
            entityset=es,
            target_dataframe_name="dataset",
            max_depth=FEATURE_ENGINEERING_DEPTH,
            verbose=False,
        )

        # Drop the index column if it appears
        feature_matrix = feature_matrix.drop(
            columns=["_index"], errors="ignore"
        )

        # Re-align index
        feature_matrix = feature_matrix.reset_index(drop=True)

        n_new = feature_matrix.shape[1] - X.shape[1]
        msg = f"Feature engineering complete. {n_new} new features added. Total: {feature_matrix.shape[1]} features."
        return feature_matrix, msg

    except Exception as e:
        return X, f"Feature engineering skipped: {e}"

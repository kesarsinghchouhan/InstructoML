"""
utils/preprocessor.py
Robust preprocessing pipeline that handles:
  - Missing values (numeric → median, categorical → mode)
  - High-cardinality text columns (dropped with warning)
  - Constant columns (dropped)
  - Datetime columns (dropped)
  - Categorical encoding (one-hot, capped at MAX_CARDINALITY unique values)
  - Returns clean X (DataFrame) and y (Series) ready for sklearn
"""

import pandas as pd
import numpy as np
from config import MAX_CARDINALITY, FILL_NUM_STRATEGY, FILL_CAT_STRATEGY, MIN_CLASS_SAMPLES, FOLD


def preprocess(df: pd.DataFrame, target: str) -> tuple:
    """
    Clean and encode df.
    Returns (X, y, report) where report is a list of info strings.
    """
    report = []
    df = df.copy()

    # ── 1. Separate target ────────────────────────────────────────────────
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    y = df[target].copy()
    X = df.drop(columns=[target])

    # ── 2. Drop rows where target is missing ─────────────────────────────
    missing_target = y.isna().sum()
    if missing_target > 0:
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        report.append(f"Dropped {missing_target} rows with missing target values.")

    # ── 2b. Drop rare classes (classification only) ───────────────────────
    # A class with fewer than FOLD samples causes CV to fail.
    # We drop those rows entirely and warn the user.
    if pd.api.types.is_object_dtype(y) or y.nunique() <= 15:
        class_counts = y.value_counts()
        rare_classes  = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()
        if rare_classes:
            mask = ~y.isin(rare_classes)
            dropped_rare = (~mask).sum()
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
            report.append(
                f"⚠️ Dropped {dropped_rare} rows belonging to rare classes "
                f"{rare_classes} (fewer than {MIN_CLASS_SAMPLES} samples each). "
                f"These classes cannot be used in {FOLD}-fold cross-validation."
            )

    # ── 3. Drop datetime columns ──────────────────────────────────────────
    dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    # also detect object cols that look like dates
    for c in X.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(X[c], errors="raise")
            dt_cols.append(c)
        except Exception:
            pass
    dt_cols = list(set(dt_cols))
    if dt_cols:
        X = X.drop(columns=dt_cols)
        report.append(f"Dropped datetime columns: {dt_cols}")

    # ── 4. Drop constant columns ──────────────────────────────────────────
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)
        report.append(f"Dropped constant columns: {const_cols}")

    # ── 5. Drop high-cardinality text columns ─────────────────────────────
    high_card = [
        c for c in X.select_dtypes(include="object").columns
        if X[c].nunique() > MAX_CARDINALITY
    ]
    if high_card:
        X = X.drop(columns=high_card)
        report.append(
            f"Dropped high-cardinality text columns (>{MAX_CARDINALITY} unique): {high_card}"
        )

    # ── 6. Fill missing numeric values ────────────────────────────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        if FILL_NUM_STRATEGY == "median":
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        else:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

    # ── 7. Fill missing categorical values ───────────────────────────────
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        for c in cat_cols:
            mode_val = X[c].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
            X[c] = X[c].fillna(fill_val)

    # ── 8. One-hot encode categoricals ───────────────────────────────────
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
        report.append(f"One-hot encoded {len(cat_cols)} categorical column(s).")

    # ── 9. Ensure all columns are numeric ────────────────────────────────
    remaining_obj = X.select_dtypes(include="object").columns.tolist()
    if remaining_obj:
        X = X.drop(columns=remaining_obj)
        report.append(f"Dropped remaining non-numeric columns: {remaining_obj}")

    # ── 10. Drop any remaining NaN / inf rows ─────────────────────────────
    X = X.replace([np.inf, -np.inf], np.nan)
    before = len(X)
    X = X.dropna()
    y = y.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    dropped = before - len(X)
    if dropped > 0:
        report.append(f"Dropped {dropped} rows with remaining NaN/Inf values.")

    if len(X) == 0:
        raise ValueError("No valid rows remain after preprocessing. Check your dataset.")

    return X, y, report

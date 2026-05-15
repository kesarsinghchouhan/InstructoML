"""
models/train.py
Training logic for Auto mode (PyCaret) and Manual mode (sklearn Pipeline).

AUTO   → run_auto_training(df, target, train_size, problem_type)
         Returns (model, results_df)

MANUAL → run_manual_training(X, y, train_size, problem_type, model_key)
         Returns (pipeline, preds, y_test, X_train, X_test, y_train)

Note: Manual mode receives already-preprocessed X (numeric DataFrame) and y.
Auto  mode receives the raw df + target (PyCaret handles its own preprocessing).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

from config import SESSION_ID, FOLD


# ── Sklearn model catalogue ───────────────────────────────────────────────────

def _get_models(problem_type: str) -> dict:
    if problem_type == "classification":
        return {
            "lr":      LogisticRegression(max_iter=1000, random_state=SESSION_ID),
            "rf":      RandomForestClassifier(random_state=SESSION_ID),
            "dt":      DecisionTreeClassifier(random_state=SESSION_ID),
            "knn":     KNeighborsClassifier(),
            "svm":     SVC(probability=True),
            "nb":      GaussianNB(),
            "xgboost": XGBClassifier(
                eval_metric="logloss", verbosity=0,
                use_label_encoder=False, random_state=SESSION_ID
            ),
        }
    return {
        "lr":      LinearRegression(),
        "rf":      RandomForestRegressor(random_state=SESSION_ID),
        "dt":      DecisionTreeRegressor(random_state=SESSION_ID),
        "knn":     KNeighborsRegressor(),
        "svm":     SVR(),
        "xgboost": XGBRegressor(verbosity=0, random_state=SESSION_ID),
    }


def _safe_fold(y: pd.Series, requested_fold: int) -> int:
    """
    Return the largest fold count that is safe for y.
    For classification: fold <= min class count.
    For regression: fold <= total rows.
    """
    min_class = int(y.value_counts().min()) if y.nunique() <= 50 else len(y)
    return max(2, min(requested_fold, min_class))


# ── AUTO MODE ─────────────────────────────────────────────────────────────────

def run_auto_training(df: pd.DataFrame, target: str,
                      train_size: float, problem_type: str):
    """
    Use PyCaret to benchmark all algorithms.
    Returns (best_model, leaderboard_df).
    """
    if problem_type == "classification":
        from pycaret.classification import (
            setup as _setup, compare_models as _compare, pull as _pull
        )
    else:
        from pycaret.regression import (
            setup as _setup, compare_models as _compare, pull as _pull
        )

    # Auto-reduce fold so no class has fewer samples than fold count
    safe = _safe_fold(df[target], FOLD)

    _setup(
        data=df,
        target=target,
        session_id=SESSION_ID,
        fold=safe,
        train_size=train_size,
        verbose=False,
        html=False,
    )
    best = _compare(verbose=False)
    leaderboard = _pull()
    return best, leaderboard


# ── MANUAL MODE ───────────────────────────────────────────────────────────────

def run_manual_training(X: pd.DataFrame, y: pd.Series,
                        train_size: float, problem_type: str,
                        model_key: str):
    """
    Train a single sklearn estimator with a proper train/test split.
    X must already be fully numeric (output of preprocessor.preprocess).
    Returns (pipeline, preds, y_test, X_train, X_test, y_train).
    """
    models = _get_models(problem_type)
    if model_key not in models:
        raise ValueError(
            f"Unknown model key '{model_key}'. Available: {list(models.keys())}"
        )

    # Use stratify for classification to preserve class distribution in splits
    stratify = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=SESSION_ID,
        stratify=stratify,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model",  models[model_key]),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    return pipeline, preds, y_test, X_train, X_test, y_train

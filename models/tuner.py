"""
models/tuner.py
Fast Optuna hyperparameter tuning.

Speed improvements over naive Optuna:
  1. MedianPruner  -> kills unpromising trials after the 2nd CV fold
  2. n_jobs=-1     -> parallel CV folds where the estimator allows
  3. Fewer trials  -> OPTUNA_TRIALS = 20 (configurable in config.py)
  4. Timeout       -> hard stop at OPTUNA_TIMEOUT seconds
  5. Smaller CV    -> OPTUNA_CV_FOLDS = 3 during HPO

Public API
----------
tune_with_optuna(X_train, y_train, X_test, model_key, problem_type)
    → (best_pipeline, best_score, best_params, tuned_preds)

PYCARET_TO_INTERNAL  ->> dict mapping PyCaret model IDs to our model keys
"""

import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

from config import SESSION_ID, OPTUNA_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_TIMEOUT

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── PyCaret abbreviation → internal key ──────────────────────────────────────
PYCARET_TO_INTERNAL = {
    "lr": "lr", "ridge": "lr", "lasso": "lr", "en": "lr",
    "lar": "lr", "llar": "lr", "omp": "lr", "br": "lr",
    "ard": "lr", "par": "lr", "ransac": "lr", "tr": "lr",
    "huber": "lr", "lda": "lr",
    "rf": "rf", "et": "rf",
    "dt": "dt", "ada": "dt",
    "knn": "knn",
    "svm": "svm", "kr": "svm",
    "nb": "nb", "qda": "nb",
    "xgboost": "xgboost", "gbc": "xgboost", "gbr": "xgboost",
    "lightgbm": "xgboost", "catboost": "xgboost",
}


# ── Search spaces ─────────────────────────────────────────────────────────────

def _build_clf(trial, key):
    if key == "lr":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            max_iter=1000, random_state=SESSION_ID
        )
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            random_state=SESSION_ID, n_jobs=-1
        )
    if key == "dt":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            random_state=SESSION_ID
        )
    if key == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            n_jobs=-1
        )
    if key == "svm":
        return SVC(
            C=trial.suggest_float("C", 1e-2, 10.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
            probability=True
        )
    if key == "nb":
        return GaussianNB(
            var_smoothing=trial.suggest_float("var_smoothing", 1e-10, 1e-7, log=True)
        )
    if key == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False, random_state=SESSION_ID,
            n_jobs=-1
        )
    raise ValueError(f"Unknown classification key: '{key}'")


def _build_reg(trial, key):
    if key == "lr":
        return LinearRegression()
    if key == "rf":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            random_state=SESSION_ID, n_jobs=-1
        )
    if key == "dt":
        return DecisionTreeRegressor(
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            random_state=SESSION_ID
        )
    if key == "knn":
        return KNeighborsRegressor(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            n_jobs=-1
        )
    if key == "svm":
        return SVR(
            C=trial.suggest_float("C", 1e-2, 10.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["rbf", "linear"])
        )
    if key == "xgboost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            verbosity=0, random_state=SESSION_ID, n_jobs=-1
        )
    raise ValueError(f"Unknown regression key: '{key}'")


# ── Public tuning function ────────────────────────────────────────────────────

def tune_with_optuna(X_train, y_train, X_test,
                     model_key: str, problem_type: str):
    """
    Fast Optuna tuning with pruning + timeout.

    Returns
    -------
    best_pipeline : fitted sklearn Pipeline
    best_score    : best mean CV score
    best_params   : dict of best hyperparameters
    tuned_preds   : predictions on X_test
    """
    scoring = "accuracy" if problem_type == "classification" else "r2"

    # MedianPruner cuts unpromising trials early 
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=SESSION_ID)
    study   = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=sampler
    )

    def objective(trial):
        if problem_type == "classification":
            model = _build_clf(trial, model_key)
        else:
            model = _build_reg(trial, model_key)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model",  model),
        ])

        # Report intermediate scores for pruning
        from sklearn.model_selection import StratifiedKFold, KFold
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                                 random_state=SESSION_ID)
        else:
            cv = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                       random_state=SESSION_ID)

        scores = []
        for step, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            from sklearn.metrics import accuracy_score, r2_score
            if problem_type == "classification":
                s = accuracy_score(y_val, pipe.predict(X_val))
            else:
                s = r2_score(y_val, pipe.predict(X_val))
            scores.append(s)
            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )

   #pipeline with best hyperparameters
    best_params = study.best_params
    fixed = optuna.trial.FixedTrial(best_params)

    if problem_type == "classification":
        best_model = _build_clf(fixed, model_key)
    else:
        best_model = _build_reg(fixed, model_key)

    best_pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model",  best_model),
    ])
    best_pipeline.fit(X_train, y_train)
    tuned_preds = best_pipeline.predict(X_test)

    return best_pipeline, study.best_value, best_params, tuned_preds

"""
app.py ->>InstructoML main application
======================================
Tab 1 │ Data       ->> upload, preview, target selection, problem detection
Tab 2 │ EDA & FE   ->> profiling report + feature engineering
Tab 3 │ Training   ->> Auto (PyCaret) or Manual (sklearn)
Tab 4 │ Results    ->> metrics, Optuna tuning, model download
"""

import io
import os
import pickle
import importlib

import pandas as pd
import streamlit as st

import config
from utils.loader          import load_csv
from utils.detect_problem  import detect_problem_type
from utils.preprocessor    import preprocess
from utils.metrics         import show_metrics
from models.train          import run_auto_training, run_manual_training

# Hyperparameter-tuning has a hyphen → must use importlib
_tuner              = importlib.import_module("models.tuner")
tune_with_optuna    = _tuner.tune_with_optuna
PYCARET_TO_INTERNAL = _tuner.PYCARET_TO_INTERNAL

os.makedirs("outputs", exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InstructoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_bytes(model) -> bytes:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.getvalue()


def _download_btn(model, label="⬇️ Download Trained Model (.pkl)"):
    st.download_button(
        label=label,
        data=_model_bytes(model),
        file_name="model.pkl",
        mime="application/octet-stream",
        type="primary",
    )


def _reset_training():
    """Clear all training/tuning state when a new dataset or target is set."""
    for key in [
        "model", "base_model", "mode", "problem_type", "target",
        "df_clean", "X", "y", "train_size",
        "results_df", "preds", "y_test", "X_train", "X_test", "y_train",
        "selected_model_key", "tuned", "tuned_preds", "best_params", "best_score",
        "preprocess_report",
    ]:
        st.session_state.pop(key, None)



# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️  InstructoML")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

st.title("🤖 InstructoML: AutoML Made Simple")

if not uploaded:
    st.info("  Upload a CSV file from the sidebar to get started.")
    st.stop()

# ── Load CSV  ─────────────────────────────────────────────────────────
df_raw = None
try:
    df_raw, enc, enc_warn = load_csv(uploaded)
except ValueError as e:
    st.error(str(e))
    st.stop()

if enc_warn:
    st.sidebar.warning(f"⚠️ {enc_warn}")


# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Data", "🔍 EDA & FE", "🏋️ Training", "📊 Results"]
)


# TAB 1 – DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(100), use_container_width=True)

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Rows",    df_raw.shape[0])
    col_info2.metric("Columns", df_raw.shape[1])
    col_info3.metric("Missing values", int(df_raw.isna().sum().sum()))

    st.divider()

    prev_target = st.session_state.get("target")

    target = st.selectbox(" Select Target Column", df_raw.columns)
    if target is None:
        st.stop()

    # Reset state if target changed
    if prev_target and prev_target != target:
        _reset_training()

    problem_type = detect_problem_type(df_raw, target)
    st.info(f" Detected problem type: **{problem_type}**")

    

    split = st.slider(
        "Train Size (%)",
        int(config.MIN_TRAIN_SIZE * 100),
        int(config.MAX_TRAIN_SIZE * 100),
        int(config.DEFAULT_TRAIN_SIZE * 100),
        int(config.STEP_TRAIN_SIZE * 100),
    )
    train_size = split / 100
    st.caption(f"Training on **{split}%**  ·  Testing on **{100 - split}%**")

    st.divider()
    st.subheader("Column Info")
    col_info = pd.DataFrame({
        "dtype":   df_raw.dtypes.astype(str),
        "missing": df_raw.isna().sum(),
        "unique":  df_raw.nunique(),
    })
    st.dataframe(col_info, use_container_width=True)

    from sklearn.preprocessing import LabelEncoder

    if df_raw[target].dtype == "object":
        le = LabelEncoder()
        df_raw[target] = le.fit_transform(df_raw[target])



# TAB 2 – EDA & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")

    if st.button("📊 Generate EDA Report"):
        with st.spinner("Generating EDA report (this may take a minute)…"):
            try:
                from utils.EDA import generate_eda
                path = generate_eda(df_raw)
                st.success(" EDA report generated.")
                with open(path, "rb") as f:
                    st.download_button(
                        " Download EDA Report",
                        f,
                        file_name="eda_report.html",
                        mime="text/html",
                    )
            except Exception as e:
                st.error(f"EDA failed: {e}")

    st.divider()
    st.subheader("Automated Feature Engineering (Featuretools)")
    st.caption(
        "Runs Deep Feature Synthesis on the preprocessed numeric features. "
        "The enriched dataset will be used for training."
    )

    if st.button(" Run Feature Engineering"):
        with st.spinner("Running feature engineering…"):
            try:
                X_tmp, y_tmp, _ = preprocess(df_raw, target)
                from utils.feature_engineering import run_feature_engineering
                X_fe, msg = run_feature_engineering(X_tmp)
                st.session_state["X_fe"] = X_fe
                st.session_state["y_fe"] = y_tmp
                st.success(msg)
                st.dataframe(X_fe.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"Feature engineering failed: {e}")



# TAB 3 – TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Training Configuration")

    mode = st.radio("Training Mode", ["auto", "manual"], horizontal=True)

    model_dict = (
        config.CLASSIFICATION_MODELS
        if problem_type == "classification"
        else config.REGRESSION_MODELS
    )

    selected_model_key = None
    if mode == "manual":
        st.markdown("#### Select Algorithm")
        selected_name      = st.selectbox("Algorithm", list(model_dict.keys()))
        if selected_name:
            selected_model_key = model_dict[selected_name]
            st.info(f"Selected: **{selected_name}**")

    st.divider()

    if st.button("Train Model", type="primary"):

        # ── Preprocess ───────────────────────────────────────────────────
        with st.spinner("Preprocessing data…"):
            try:
                # Use FE result if available, else preprocess raw
                if "X_fe" in st.session_state:
                    X_clean = st.session_state["X_fe"]
                    y_clean = st.session_state["y_fe"]
                    prep_report = ["Using feature-engineered dataset."]
                else:
                    X_clean, y_clean, prep_report = preprocess(df_raw, target)
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.stop()

        # Show preprocessing report
        if prep_report:
            with st.expander("Preprocessing report", expanded=False):
                for msg in prep_report:
                    st.write(f"• {msg}")

        # ── Train ────────────────────────────────────────────────────────
        with st.spinner("Training in progress… please wait."):
            try:
                if mode == "auto":
                    # PyCaret needs the full df with target column
                    df_for_pycaret = X_clean.copy()
                    df_for_pycaret[target] = y_clean.values
                    model, results_df = run_auto_training(
                        df_for_pycaret, target, train_size, problem_type
                    )
                    st.session_state.update({
                        "model":        model,
                        "base_model":   model,
                        "results_df":   results_df,
                        "preds":        None,
                        "y_test":       None,
                        "X_train":      None,
                        "X_test":       None,
                        "y_train":      None,
                    })

                else:
                    if selected_model_key is None:
                        st.error("No model selected. Please select an algorithm.")
                        st.stop()
                    pipeline, preds, y_test, X_train, X_test, y_train = (
                        run_manual_training(
                            X_clean, y_clean, train_size,
                            problem_type, selected_model_key
                        )
                    )
                    st.session_state.update({
                        "model":        pipeline,
                        "base_model":   pipeline,
                        "results_df":   None,
                        "preds":        preds,
                        "y_test":       y_test,
                        "X_train":      X_train,
                        "X_test":       X_test,
                        "y_train":      y_train,
                    })

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

        # Persist shared state
        st.session_state.update({
            "mode":               mode,
            "problem_type":       problem_type,
            "target":             target,
            "train_size":         train_size,
            "X":                  X_clean,
            "y":                  y_clean,
            "selected_model_key": selected_model_key,
            "tuned":              False,
            "tuned_preds":        None,
            "best_params":        None,
            "best_score":         None,
        })

        st.success("✅ Training complete! Head to the **Results** tab.")



# TAB 4 – RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:

    if "model" not in st.session_state:
        st.warning("⚠️ Run training first (Tab 3).")
        st.stop()

    s_mode    = st.session_state["mode"]
    s_pt      = st.session_state["problem_type"]
    tuned     = st.session_state.get("tuned", False)

    # ──────────────────────────────────────────────────────────────────────
    # AUTO MODE
    # ──────────────────────────────────────────────────────────────────────
    if s_mode == "auto":

        st.subheader("📊 Model Comparison — All Algorithms")
        st.dataframe(
            st.session_state["results_df"], use_container_width=True
        )
        st.caption("Ranked by primary metric. Top row = best model.")
        st.divider()

        if not tuned:
            st.markdown("####  Hyperparameter Tuning (Optuna)")
            st.caption(
                f"Runs up to **{config.OPTUNA_TRIALS} trials** with "
                f"**{config.OPTUNA_CV_FOLDS}-fold CV** and a "
                f"**{config.OPTUNA_TIMEOUT}s timeout** — typically 1–2 minutes."
            )
            tune_choice = st.radio(
                "Tune the best model with Optuna?",
                ["No", "Yes"], key="auto_tune_radio"
            )

            if tune_choice == "No":
                st.info("Tuning skipped.")
                _download_btn(st.session_state["model"])

            else:
                if st.button("⚡ Run Optuna Tuning", key="auto_tune_btn"):

                    # Identify best model internal key
                    best_row    = str(st.session_state["results_df"].index[0]).lower()
                    internal_key = PYCARET_TO_INTERNAL.get(best_row, "rf")

                    # Build train/test split from preprocessed data
                    from sklearn.model_selection import train_test_split
                    X_all  = st.session_state["X"]
                    y_all  = st.session_state["y"]
                    ts     = st.session_state["train_size"]
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_all, y_all, train_size=ts,
                        random_state=config.SESSION_ID
                    )

                    with st.spinner(
                        f"Tuning **{internal_key}** with Optuna "
                        f"({config.OPTUNA_TRIALS} trials, max {config.OPTUNA_TIMEOUT}s)…"
                    ):
                        try:
                            best_pipe, best_score, best_params, tuned_preds = (
                                tune_with_optuna(X_tr, y_tr, X_te, internal_key, s_pt)
                            )
                        except Exception as e:
                            st.error(f"Tuning failed: {e}")
                            st.stop()

                    st.session_state.update({
                        "model":       best_pipe,
                        "tuned":       True,
                        "tuned_preds": tuned_preds,
                        "y_test":      y_te,
                        "best_score":  best_score,
                        "best_params": best_params,
                    })
                    st.rerun()

        if tuned:
            st.subheader("After Optuna Tuning")
            c1, c2 = st.columns(2)
            c1.metric("Best CV Score", f"{st.session_state['best_score']:.4f}")
            with c2:
                st.markdown("**Best Hyperparameters**")
                st.json(st.session_state["best_params"])

            st.markdown("#### Test-set Evaluation")
            show_metrics(
                st.session_state["y_test"],
                st.session_state["tuned_preds"],
                s_pt,
            )
            st.divider()
            _download_btn(st.session_state["model"])

    
    # MANUAL MODE
    # ──────────────────────────────────────────────────────────────────────
    else:

        st.subheader(" Evaluation — Before Tuning")
        show_metrics(
            st.session_state["y_test"],
            st.session_state["preds"],
            s_pt,
        )
        st.divider()

        if not tuned:
            st.markdown("####  Hyperparameter Tuning (Optuna)")
            st.caption(
                f"Runs up to **{config.OPTUNA_TRIALS} trials** with "
                f"**{config.OPTUNA_CV_FOLDS}-fold CV** and a "
                f"**{config.OPTUNA_TIMEOUT}s timeout** — typically under 2 minutes."
            )
            tune_choice = st.radio(
                "Tune this model with Optuna?",
                ["No", "Yes"], key="manual_tune_radio"
            )

            if tune_choice == "No":
                st.info("Tuning skipped.")
                _download_btn(st.session_state["model"])

            else:
                if st.button(" Run Optuna Tuning", key="manual_tune_btn"):
                    mk = st.session_state["selected_model_key"]
                    with st.spinner(
                        f"Tuning **{mk}** with Optuna "
                        f"({config.OPTUNA_TRIALS} trials, max {config.OPTUNA_TIMEOUT}s)…"
                    ):
                        try:
                            best_pipe, best_score, best_params, tuned_preds = (
                                tune_with_optuna(
                                    st.session_state["X_train"],
                                    st.session_state["y_train"],
                                    st.session_state["X_test"],
                                    mk, s_pt,
                                )
                            )
                        except Exception as e:
                            st.error(f"Tuning failed: {e}")
                            st.stop()

                    st.session_state.update({
                        "model":       best_pipe,
                        "tuned":       True,
                        "tuned_preds": tuned_preds,
                        "best_score":  best_score,
                        "best_params": best_params,
                    })
                    st.rerun()

        if tuned:
            st.subheader(" Evaluation — After Optuna Tuning")
            c1, c2 = st.columns(2)
            c1.metric("Best CV Score", f"{st.session_state['best_score']:.4f}")
            with c2:
                st.markdown("**Best Hyperparameters**")
                st.json(st.session_state["best_params"])

            st.markdown("#### Test-set Evaluation")
            show_metrics(
                st.session_state["y_test"],
                st.session_state["tuned_preds"],
                s_pt,
            )
            st.divider()
            _download_btn(st.session_state["model"])

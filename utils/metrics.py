"""
utils/metrics.py
Render evaluation metrics in Streamlit for classification and regression.
"""

import pandas as pd
import streamlit as st


def show_metrics(y, preds, problem_type: str):
    if problem_type == "classification":
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report, confusion_matrix
        )
        acc = accuracy_score(y, preds)
        f1  = f1_score(y, preds, average="weighted", zero_division=0)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy",       f"{acc:.4f}")
        c2.metric("F1 (weighted)",  f"{f1:.4f}")

        st.markdown("**Classification Report**")
        st.code(classification_report(y, preds, zero_division=0))

        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y, preds)
        st.dataframe(
            pd.DataFrame(
                cm,
                index=[f"Actual {c}"    for c in sorted(set(y))],
                columns=[f"Predicted {c}" for c in sorted(set(y))],
            )
        )

    else:
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        r2   = r2_score(y, preds)
        mse  = mean_squared_error(y, preds)
        mae  = mean_absolute_error(y, preds)
        rmse = np.sqrt(mse)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MSE",      f"{mse:.4f}")
        c3.metric("MAE",      f"{mae:.4f}")
        c4.metric("RMSE",     f"{rmse:.4f}")

"""
utils/EDA.py
Generate a ydata-profiling HTML report.
"""

import os
from config import EDA_REPORT_PATH


def generate_eda(df):
    os.makedirs(os.path.dirname(EDA_REPORT_PATH), exist_ok=True)
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, explorative=True, minimal=False)
    except ImportError:
        # Fallback to pandas-profiling if ydata-profiling not installed
        from pandas_profiling import ProfileReport
        profile = ProfileReport(df, explorative=True)

    profile.to_file(EDA_REPORT_PATH)
    return EDA_REPORT_PATH

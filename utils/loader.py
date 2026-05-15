"""
utils/loader.py
Robust CSV loader — tries multiple encodings automatically.
Returns (DataFrame, encoding_used, warning_message_or_None).
"""

import pandas as pd


_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


def load_csv(file_obj) -> tuple:
    """
    Try to read a CSV with several encodings.
    Returns (df, encoding, warn) where warn is None on clean load
    or a string describing the fallback used.
    """
    last_err = None

    for enc in _ENCODINGS:
        try:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding=enc, low_memory=False)
            warn = None if enc == "utf-8" else f"File read with encoding '{enc}'."
            return df, enc, warn
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            continue

    # Last resort: replace bad bytes
    try:
        file_obj.seek(0)
        raw = file_obj.read().decode("utf-8", errors="replace")
        from io import StringIO
        df = pd.read_csv(StringIO(raw), low_memory=False)
        return df, "utf-8 (errors replaced)", "Some characters were unreadable and replaced."
    except Exception as e:
        raise ValueError(
            f"Could not read CSV file. Last error: {last_err}. "
            "Please check the file is a valid CSV."
        ) from e

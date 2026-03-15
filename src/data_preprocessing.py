"""Data preprocessing utilities for IPL score prediction."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """Load IPL dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(filepath)


def encode_categorical(df, columns=None):
    """Label-encode categorical columns in-place and return the encoders.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (will be modified in-place).
    columns : list[str] or None
        Columns to encode. If *None*, all object-type columns are encoded.

    Returns
    -------
    dict[str, LabelEncoder]
        Mapping of column name to its fitted LabelEncoder.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return encoders


def prepare_features(df, target_col="total_score"):
    """Split a DataFrame into feature matrix and target vector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all columns (including target).
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X are features and y is the target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

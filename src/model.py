"""Model training and evaluation for IPL score prediction."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target into train/test sets.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, model_type="random_forest", **kwargs):
    """Train a regression model.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    model_type : str
        Either ``"linear_regression"`` or ``"random_forest"``.
    **kwargs
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    sklearn estimator
        Fitted model.
    """
    if model_type == "linear_regression":
        model = LinearRegression(**kwargs)
    elif model_type == "random_forest":
        kwargs.setdefault("n_estimators", 100)
        kwargs.setdefault("random_state", 42)
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on test data.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, and R² metrics.
    """
    predictions = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }


def predict_score(model, features):
    """Predict total score given match-state features.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model.
    features : array-like
        2-D array of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Predicted scores.
    """
    return model.predict(np.atleast_2d(features))

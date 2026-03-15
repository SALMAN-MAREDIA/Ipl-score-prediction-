"""Tests for IPL score prediction project."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import encode_categorical, load_data, prepare_features
from src.generate_dataset import generate_dataset, IPL_TEAMS, VENUES
from src.model import evaluate_model, predict_score, split_data, train_model


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_returns_dataframe(self):
        df = generate_dataset(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = generate_dataset(n_samples=200)
        assert len(df) == 200

    def test_columns(self):
        df = generate_dataset(n_samples=10)
        expected_cols = {
            "batting_team", "bowling_team", "venue", "overs",
            "current_score", "wickets", "runs_last_5_overs",
            "wickets_last_5_overs", "total_score",
        }
        assert set(df.columns) == expected_cols

    def test_teams_are_valid(self):
        df = generate_dataset(n_samples=500)
        assert df["batting_team"].isin(IPL_TEAMS).all()
        assert df["bowling_team"].isin(IPL_TEAMS).all()

    def test_venues_are_valid(self):
        df = generate_dataset(n_samples=500)
        assert df["venue"].isin(VENUES).all()

    def test_total_score_exceeds_current_score(self):
        df = generate_dataset(n_samples=500)
        assert (df["total_score"] > df["current_score"]).all()

    def test_overs_range(self):
        df = generate_dataset(n_samples=500)
        assert df["overs"].between(5.0, 19.5).all()

    def test_wickets_range(self):
        df = generate_dataset(n_samples=500)
        assert df["wickets"].between(0, 9).all()

    def test_reproducibility(self):
        df1 = generate_dataset(n_samples=50, seed=99)
        df2 = generate_dataset(n_samples=50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    @pytest.fixture()
    def sample_df(self):
        return generate_dataset(n_samples=100)

    def test_encode_categorical_returns_encoders(self, sample_df):
        encoders = encode_categorical(sample_df)
        assert "batting_team" in encoders
        assert "bowling_team" in encoders
        assert "venue" in encoders

    def test_encode_categorical_produces_numeric(self, sample_df):
        encode_categorical(sample_df)
        assert pd.api.types.is_integer_dtype(sample_df["batting_team"])
        assert pd.api.types.is_integer_dtype(sample_df["bowling_team"])
        assert pd.api.types.is_integer_dtype(sample_df["venue"])

    def test_prepare_features_shape(self, sample_df):
        encode_categorical(sample_df)
        X, y = prepare_features(sample_df)
        assert X.shape[0] == 100
        assert X.shape[1] == 8
        assert len(y) == 100

    def test_prepare_features_target_not_in_X(self, sample_df):
        encode_categorical(sample_df)
        X, _ = prepare_features(sample_df)
        assert "total_score" not in X.columns

    def test_load_data(self, tmp_path):
        df = generate_dataset(n_samples=10)
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loaded = load_data(str(path))
        assert len(loaded) == 10


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

class TestModel:
    @pytest.fixture()
    def data(self):
        df = generate_dataset(n_samples=500, seed=42)
        encode_categorical(df)
        X, y = prepare_features(df)
        return split_data(X, y)

    def test_train_random_forest(self, data):
        X_train, _, y_train, _ = data
        model = train_model(X_train, y_train, model_type="random_forest")
        assert hasattr(model, "predict")

    def test_train_linear_regression(self, data):
        X_train, _, y_train, _ = data
        model = train_model(X_train, y_train, model_type="linear_regression")
        assert hasattr(model, "predict")

    def test_train_invalid_model_raises(self, data):
        X_train, _, y_train, _ = data
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(X_train, y_train, model_type="unknown")

    def test_evaluate_model_keys(self, data):
        X_train, X_test, y_train, y_test = data
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_evaluate_model_r2_positive(self, data):
        X_train, X_test, y_train, y_test = data
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["r2"] > 0, "R² should be positive for a reasonable model"

    def test_predict_score_shape(self, data):
        X_train, X_test, y_train, _ = data
        model = train_model(X_train, y_train)
        preds = predict_score(model, X_test.values[:5])
        assert preds.shape == (5,)

    def test_predictions_are_finite(self, data):
        X_train, X_test, y_train, _ = data
        model = train_model(X_train, y_train)
        preds = predict_score(model, X_test)
        assert np.all(np.isfinite(preds))

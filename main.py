"""IPL Score Prediction – end-to-end pipeline.

Usage
-----
    python main.py                  # train + evaluate
    python main.py --predict        # interactive prediction
"""

import argparse
import os
import sys

import numpy as np

# Add project root to path so ``src`` is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import encode_categorical, load_data, prepare_features
from src.generate_dataset import IPL_TEAMS, VENUES, generate_dataset
from src.model import evaluate_model, predict_score, split_data, train_model


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ipl_data.csv")


def ensure_dataset():
    """Generate the dataset if it does not exist yet."""
    if not os.path.exists(DATA_PATH):
        print("Dataset not found – generating synthetic data …")
        df = generate_dataset()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(df)} rows to {DATA_PATH}")


def run_training_pipeline(model_type="random_forest"):
    """Load data, preprocess, train, and evaluate.

    Returns
    -------
    tuple
        (model, encoders, metrics)
    """
    ensure_dataset()

    df = load_data(DATA_PATH)
    encoders = encode_categorical(df)
    X, y = prepare_features(df, target_col="total_score")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, model_type=model_type)
    metrics = evaluate_model(model, X_test, y_test)

    print("\n=== Model Evaluation ===")
    print(f"Model        : {model_type}")
    print(f"MAE          : {metrics['mae']:.2f}")
    print(f"RMSE         : {metrics['rmse']:.2f}")
    print(f"R² Score     : {metrics['r2']:.4f}")

    return model, encoders, metrics


def interactive_predict(model, encoders):
    """Prompt the user for match-state features and predict the total score."""
    print("\n=== IPL Score Predictor ===")
    print("Enter current match details:\n")

    batting_team = _choose("Batting team", IPL_TEAMS)
    bowling_team = _choose(
        "Bowling team", [t for t in IPL_TEAMS if t != batting_team]
    )
    venue = _choose("Venue", VENUES)
    overs = float(input("Overs completed (e.g. 12.3): "))
    current_score = int(input("Current score: "))
    wickets = int(input("Wickets fallen: "))
    runs_last_5 = int(input("Runs in last 5 overs: "))
    wickets_last_5 = int(input("Wickets in last 5 overs: "))

    bat_enc = encoders["batting_team"].transform([batting_team])[0]
    bowl_enc = encoders["bowling_team"].transform([bowling_team])[0]
    venue_enc = encoders["venue"].transform([venue])[0]

    features = np.array(
        [[bat_enc, bowl_enc, venue_enc, overs, current_score,
          wickets, runs_last_5, wickets_last_5]]
    )
    predicted = predict_score(model, features)[0]
    print(f"\n>>> Predicted Total Score: {int(round(predicted))}")


def _choose(label, options):
    """Display numbered options and return the chosen value."""
    print(f"{label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    idx = int(input(f"Select {label} (number): ")) - 1
    return options[idx]


def main():
    parser = argparse.ArgumentParser(description="IPL Score Prediction")
    parser.add_argument(
        "--predict", action="store_true", help="Run interactive prediction"
    )
    parser.add_argument(
        "--model",
        choices=["linear_regression", "random_forest"],
        default="random_forest",
        help="Model type (default: random_forest)",
    )
    args = parser.parse_args()

    model, encoders, _metrics = run_training_pipeline(model_type=args.model)

    if args.predict:
        interactive_predict(model, encoders)


if __name__ == "__main__":
    main()

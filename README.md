# IPL Score Prediction

A machine learning project that predicts the **total score** of an IPL innings
based on the current match state (overs completed, runs scored, wickets fallen,
recent run-rate, venue, and teams).

## Project Structure

```
├── data/                     # Generated dataset
│   └── ipl_data.csv
├── src/
│   ├── generate_dataset.py   # Synthetic IPL dataset generator
│   ├── data_preprocessing.py # Loading, encoding & feature prep
│   └── model.py              # Model training, evaluation & prediction
├── tests/
│   └── test_model.py         # Unit tests
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train and evaluate

```bash
python main.py                       # Random Forest (default)
python main.py --model linear_regression
```

### Interactive prediction

```bash
python main.py --predict
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Dataset

The dataset is synthetically generated to simulate realistic IPL innings
snapshots. Each row contains:

| Feature              | Description                              |
| -------------------- | ---------------------------------------- |
| `batting_team`       | Name of the batting team                 |
| `bowling_team`       | Name of the bowling team                 |
| `venue`              | Match venue                              |
| `overs`              | Overs completed (5.0 – 19.5)            |
| `current_score`      | Runs scored so far                       |
| `wickets`            | Wickets fallen                           |
| `runs_last_5_overs`  | Runs scored in the last 5 overs          |
| `wickets_last_5_overs` | Wickets fallen in the last 5 overs     |
| **`total_score`**    | **Final innings total (target)**         |

## Models

| Model              | MAE   | RMSE  | R²     |
| ------------------ | ----- | ----- | ------ |
| Random Forest      | 11.95 | 15.85 | 0.7392 |
| Linear Regression  | 11.94 | 15.35 | 0.7555 |

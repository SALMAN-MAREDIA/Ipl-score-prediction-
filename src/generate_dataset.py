"""Generate a synthetic IPL dataset for score prediction."""

import random

import numpy as np
import pandas as pd

IPL_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
]

VENUES = [
    "Wankhede Stadium",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "Feroz Shah Kotla",
    "Rajiv Gandhi International Stadium",
    "MA Chidambaram Stadium",
    "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium",
    "Narendra Modi Stadium",
    "Arun Jaitley Stadium",
]


def generate_dataset(n_samples=5000, seed=42):
    """Generate a synthetic IPL match dataset.

    Each row represents the state of an innings at a given point (after
    a certain number of overs) and includes the final total score as the
    target variable.

    Parameters
    ----------
    n_samples : int
        Number of rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with match features and target score.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    records = []
    for _ in range(n_samples):
        batting_team = random.choice(IPL_TEAMS)
        bowling_team = random.choice([t for t in IPL_TEAMS if t != batting_team])
        venue = random.choice(VENUES)

        # Overs completed so far (between 5.0 and 19.5 in 0.1 increments)
        overs = round(rng.uniform(5.0, 19.5), 1)

        # Simulate current match state
        run_rate = rng.uniform(5.0, 12.0)
        current_score = int(overs * run_rate)
        wickets = min(rng.poisson(lam=2.5), 9)

        # Runs and wickets in last 5 overs
        last5_overs = min(overs, 5.0)
        last5_run_rate = rng.uniform(5.0, 14.0)
        runs_last_5 = int(last5_overs * last5_run_rate)
        wickets_last_5 = min(rng.poisson(lam=1.0), min(wickets, 5))

        # Remaining overs
        remaining_overs = round(20.0 - overs, 1)

        # Final total score (target)
        projected_additional = remaining_overs * rng.uniform(6.0, 13.0)
        # Adjust based on wickets – more wickets lower the projection
        wicket_factor = max(0.4, 1.0 - wickets * 0.06)
        total_score = int(current_score + projected_additional * wicket_factor)
        total_score = max(current_score + 1, total_score)

        records.append(
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "overs": overs,
                "current_score": current_score,
                "wickets": wickets,
                "runs_last_5_overs": runs_last_5,
                "wickets_last_5_overs": wickets_last_5,
                "total_score": total_score,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    import os

    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "ipl_data.csv")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}  ({len(df)} rows)")

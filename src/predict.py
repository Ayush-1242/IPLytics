import json
import joblib
import pandas as pd

from pathlib import Path

# --------------------------------------------------
# VENUE NAME STANDARDIZATION
# --------------------------------------------------

VENUE_ALIASES = {
    "M Chinnaswamy Stadium, Bangalore":
        "M Chinnaswamy Stadium, Bengaluru",

    "Feroz Shah Kotla, Delhi":
        "Arun Jaitley Stadium",

    "Punjab Cricket Association Stadium, Mohali, Chandigarh":
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",

    "Sardar Patel Stadium, Motera, Ahmedabad":
        "Narendra Modi Stadium, Ahmedabad"
}


def clean_venue_name(venue):
    return VENUE_ALIASES.get(venue, venue)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MATCHES_PATH = ROOT / "data" / "processed" / "matches.csv"
SELECTION_PATH = ROOT / "models" / "model_selection.json"
MODELS_DIR = ROOT / "models"


# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

CATEGORICAL_FEATURES = [
    "team1",
    "team2",
    "venue",
    "city"
]

NUMERIC_FEATURES = [
    "season",
    "team1_win_pct",
    "team2_win_pct",
    "team1_matches_played",
    "team2_matches_played",
    "team1_last5_win_pct",
    "team2_last5_win_pct",
    "h2h_team1_win_pct",
    "h2h_matches",
    "team1_venue_win_pct",
    "team2_venue_win_pct",
    "team1_venue_matches",
    "team2_venue_matches",
    "win_pct_difference",
    "last5_win_pct_difference",
    "venue_win_pct_difference",
    "h2h_advantage",
    "matches_played_difference",
    "venue_matches_difference",
    "team1_squad_batting_average",
    "team2_squad_batting_average",
    "team1_squad_batting_strike_rate",
    "team2_squad_batting_strike_rate",
    "team1_squad_bowling_economy",
    "team2_squad_bowling_economy",
    "team1_squad_bowling_strike_rate",
    "team2_squad_bowling_strike_rate",
    "team1_squad_player_count",
    "team2_squad_player_count",
    "squad_batting_average_difference",
    "squad_batting_strike_rate_difference",
    "squad_bowling_economy_difference",
    "squad_bowling_strike_rate_difference",
    "team1_elo",
    "team2_elo",
    "elo_difference",
    "team1_last10_win_pct",
    "team2_last10_win_pct",
    "last10_win_pct_difference"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------

features_df = pd.read_csv(FEATURES_PATH)
matches_df = pd.read_csv(MATCHES_PATH)

features_df["date"] = pd.to_datetime(features_df["date"])
matches_df["date"] = pd.to_datetime(matches_df["date"])


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

def load_selected_model():

    with open(SELECTION_PATH, "r") as file:
        selection = json.load(file)

    selected_name = selection["selected_model"]

    model_files = {
        "Logistic Regression": "logistic_regression_final.joblib",
        "Random Forest": "random_forest_final.joblib",
        "XGBoost": "xgboost_final.joblib"
    }

    if selected_name not in model_files:
        raise ValueError(
            f"Unknown selected model: {selected_name}"
        )

    model_path = MODELS_DIR / model_files[selected_name]

    model = joblib.load(model_path)

    return model, selected_name


# --------------------------------------------------
# GET LATEST HISTORICAL TEAM STATISTICS
# --------------------------------------------------

def get_team_statistics(team, prediction_date):

    historical = features_df[
        (features_df["date"] < prediction_date) &
        (
            (features_df["team1"] == team) |
            (features_df["team2"] == team)
        )
    ].sort_values("date")

    if historical.empty:
        raise ValueError(
            f"No historical data available for {team}"
        )

    latest = historical.iloc[-1]

    if latest["team1"] == team:
        prefix = "team1"
    else:
        prefix = "team2"

    statistics = {
        "win_pct": latest[f"{prefix}_win_pct"],
        "matches_played": latest[f"{prefix}_matches_played"],
        "last5_win_pct": latest[f"{prefix}_last5_win_pct"],
        "last10_win_pct": latest[f"{prefix}_last10_win_pct"],
        "elo": latest[f"{prefix}_elo"],
        "squad_batting_average":
            latest[f"{prefix}_squad_batting_average"],
        "squad_batting_strike_rate":
            latest[f"{prefix}_squad_batting_strike_rate"],
        "squad_bowling_economy":
            latest[f"{prefix}_squad_bowling_economy"],
        "squad_bowling_strike_rate":
            latest[f"{prefix}_squad_bowling_strike_rate"],
        "squad_player_count":
            latest[f"{prefix}_squad_player_count"]
    }

    return statistics


# --------------------------------------------------
# HEAD-TO-HEAD STATISTICS
# --------------------------------------------------

def get_head_to_head(team1, team2, prediction_date):

    history = matches_df[
        (matches_df["date"] < prediction_date) &
        (
            (
                (matches_df["team1"] == team1) &
                (matches_df["team2"] == team2)
            ) |
            (
                (matches_df["team1"] == team2) &
                (matches_df["team2"] == team1)
            )
        )
    ]

    total = len(history)

    if total == 0:
        return 0.5, 0

    team1_wins = (history["winner"] == team1).sum()

    return team1_wins / total, total


# --------------------------------------------------
# VENUE STATISTICS
# --------------------------------------------------

def get_venue_statistics(team, venue, prediction_date):

    history = matches_df[
        (matches_df["date"] < prediction_date) &
        (matches_df["venue"].apply(clean_venue_name) == venue) &
        (
            (matches_df["team1"] == team) |
            (matches_df["team2"] == team)
        )
    ]

    total = len(history)

    if total == 0:
        return 0.5, 0

    wins = (history["winner"] == team).sum()

    return wins / total, total


# --------------------------------------------------
# CREATE UPCOMING MATCH FEATURES
# --------------------------------------------------

def create_match_features(
    team1,
    team2,
    venue,
    city,
    prediction_date
):

    prediction_date = pd.Timestamp(prediction_date)

    if team1 == team2:
        raise ValueError("Please select two different teams.")

    t1 = get_team_statistics(team1, prediction_date)
    t2 = get_team_statistics(team2, prediction_date)

    h2h_pct, h2h_matches = get_head_to_head(
        team1, team2, prediction_date
    )

    venue1_pct, venue1_matches = get_venue_statistics(
        team1, venue, prediction_date
    )

    venue2_pct, venue2_matches = get_venue_statistics(
        team2, venue, prediction_date
    )

    data = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,

        "season": prediction_date.year,

        "team1_win_pct": t1["win_pct"],
        "team2_win_pct": t2["win_pct"],

        "team1_matches_played": t1["matches_played"],
        "team2_matches_played": t2["matches_played"],

        "team1_last5_win_pct": t1["last5_win_pct"],
        "team2_last5_win_pct": t2["last5_win_pct"],

        "h2h_team1_win_pct": h2h_pct,
        "h2h_matches": h2h_matches,

        "team1_venue_win_pct": venue1_pct,
        "team2_venue_win_pct": venue2_pct,

        "team1_venue_matches": venue1_matches,
        "team2_venue_matches": venue2_matches,

        "win_pct_difference":
            t1["win_pct"] - t2["win_pct"],

        "last5_win_pct_difference":
            t1["last5_win_pct"] - t2["last5_win_pct"],

        "venue_win_pct_difference":
            venue1_pct - venue2_pct,

        "h2h_advantage": h2h_pct - 0.5,

        "matches_played_difference":
            t1["matches_played"] - t2["matches_played"],

        "venue_matches_difference":
            venue1_matches - venue2_matches,

        "team1_squad_batting_average":
            t1["squad_batting_average"],

        "team2_squad_batting_average":
            t2["squad_batting_average"],

        "team1_squad_batting_strike_rate":
            t1["squad_batting_strike_rate"],

        "team2_squad_batting_strike_rate":
            t2["squad_batting_strike_rate"],

        "team1_squad_bowling_economy":
            t1["squad_bowling_economy"],

        "team2_squad_bowling_economy":
            t2["squad_bowling_economy"],

        "team1_squad_bowling_strike_rate":
            t1["squad_bowling_strike_rate"],

        "team2_squad_bowling_strike_rate":
            t2["squad_bowling_strike_rate"],

        "team1_squad_player_count":
            t1["squad_player_count"],

        "team2_squad_player_count":
            t2["squad_player_count"],

        "squad_batting_average_difference":
            t1["squad_batting_average"] -
            t2["squad_batting_average"],

        "squad_batting_strike_rate_difference":
            t1["squad_batting_strike_rate"] -
            t2["squad_batting_strike_rate"],

        "squad_bowling_economy_difference":
            t1["squad_bowling_economy"] -
            t2["squad_bowling_economy"],

        "squad_bowling_strike_rate_difference":
            t1["squad_bowling_strike_rate"] -
            t2["squad_bowling_strike_rate"],

        "team1_elo": t1["elo"],
        "team2_elo": t2["elo"],

        "elo_difference": t1["elo"] - t2["elo"],

        "team1_last10_win_pct": t1["last10_win_pct"],
        "team2_last10_win_pct": t2["last10_win_pct"],

        "last10_win_pct_difference":
            t1["last10_win_pct"] - t2["last10_win_pct"]
    }

    match_df = pd.DataFrame([data])

    return match_df[FEATURE_COLUMNS]


# --------------------------------------------------
# PREDICT MATCH WINNER
# --------------------------------------------------

def predict_match(
    team1,
    team2,
    venue,
    city,
    prediction_date
):

    model, model_name = load_selected_model()

    match_features = create_match_features(
        team1,
        team2,
        venue,
        city,
        prediction_date
    )

    probabilities = model.predict_proba(match_features)[0]

    classes = list(model.classes_)

    team1_probability = probabilities[classes.index(1)]
    team2_probability = probabilities[classes.index(0)]

    predicted_winner = (
        team1 if team1_probability >= team2_probability
        else team2
    )

    return {
        "selected_model": model_name,
        "team1": team1,
        "team2": team2,
        "predicted_winner": predicted_winner,
        "team1_probability": round(
            float(team1_probability) * 100, 2
        ),
        "team2_probability": round(
            float(team2_probability) * 100, 2
        ),
        "prediction_date": str(prediction_date)
    }


# --------------------------------------------------
# AVAILABLE TEAMS AND VENUES
# --------------------------------------------------

def get_available_teams():

    teams = set(features_df["team1"].dropna())
    teams.update(features_df["team2"].dropna())

    return sorted(teams)


def get_available_venues():

    venues = (
        features_df["venue"]
        .dropna()
        .apply(clean_venue_name)
        .unique()
        .tolist()
    )

    return sorted(venues)


# --------------------------------------------------
# TEST FROM COMMAND LINE
# --------------------------------------------------

if __name__ == "__main__":

    print("\nIPLytics - IPL Winner Predictor")
    print("=" * 40)

    teams = get_available_teams()

    print("\nAvailable teams:")

    for team in teams:
        print("-", team)

    print("\nAvailable venues:")

    venues = get_available_venues()

    for venue in venues:
        print("-", venue)

    print("\nUse the predict_match() function to make a prediction.")

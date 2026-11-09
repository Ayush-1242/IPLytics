import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
print("Loading dataset...")

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


TARGET = "team1_win"

categorical_features = [
    "team1",
    "team2",
    "venue",
    "city"
]

numeric_features = [
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
    "last10_win_pct_difference",
]

FEATURES = categorical_features + numeric_features


# Define team-paired features
swap_pairs = [
    ("team1_win_pct", "team2_win_pct"),
    ("team1_matches_played", "team2_matches_played"),
    ("team1_last5_win_pct", "team2_last5_win_pct"),
    ("team1_venue_win_pct", "team2_venue_win_pct"),
    ("team1_venue_matches", "team2_venue_matches"),
    ("team1_squad_batting_average", "team2_squad_batting_average"),
    ("team1_squad_batting_strike_rate", "team2_squad_batting_strike_rate"),
    ("team1_squad_bowling_economy", "team2_squad_bowling_economy"),
    ("team1_squad_bowling_strike_rate", "team2_squad_bowling_strike_rate"),
    ("team1_squad_player_count", "team2_squad_player_count"),
    ("team1_elo", "team2_elo"),
    ("team1_last10_win_pct", "team2_last10_win_pct"),
]

difference_features = [
    "win_pct_difference",
    "last5_win_pct_difference",
    "venue_win_pct_difference",
    "h2h_advantage",
    "matches_played_difference",
    "venue_matches_difference",
    "squad_batting_average_difference",
    "squad_batting_strike_rate_difference",
    "squad_bowling_economy_difference",
    "squad_bowling_strike_rate_difference",
    "elo_difference",
    "last10_win_pct_difference",
]


# Create reversed team-order examples
def swap_teams(X):

    swapped = X.copy()

    # Swap team names
    swapped["team1"], swapped["team2"] = (
        X["team2"].copy(),
        X["team1"].copy()
    )

    # Swap paired statistics
    for col1, col2 in swap_pairs:
        swapped[col1] = X[col2].copy()
        swapped[col2] = X[col1].copy()

    # Reverse relative differences
    for col in difference_features:
        swapped[col] = -X[col]

    # Reverse head-to-head win percentage
    swapped["h2h_team1_win_pct"] = (
        1 - X["h2h_team1_win_pct"]
    )

    return swapped


# Create model pipeline
def create_model():

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])


# Walk-forward validation
results = []

test_seasons = [
    season for season in sorted(df["season"].dropna().unique())
    if season >= 2019
]

print("\nStarting team-swapping validation...")

for season in test_seasons:

    train_mask = df["season"] < season
    test_mask = df["season"] == season

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]

    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    print("\n" + "=" * 55)
    print("Testing season:", int(season))

    # Original model
    original_model = create_model()
    original_model.fit(X_train, y_train)

    original_predictions = original_model.predict(X_test)
    original_probabilities = original_model.predict_proba(X_test)[:, 1]

    # Augmented training data only
    X_reversed = swap_teams(X_train)
    y_reversed = 1 - y_train

    X_augmented = pd.concat(
        [X_train, X_reversed],
        ignore_index=True
    )

    y_augmented = pd.concat(
        [y_train, y_reversed],
        ignore_index=True
    )

    # Augmented model
    augmented_model = create_model()
    augmented_model.fit(X_augmented, y_augmented)

    augmented_predictions = augmented_model.predict(X_test)
    augmented_probabilities = augmented_model.predict_proba(X_test)[:, 1]

    # Evaluate both models
    original_accuracy = accuracy_score(
        y_test, original_predictions
    )

    augmented_accuracy = accuracy_score(
        y_test, augmented_predictions
    )

    original_auc = roc_auc_score(
        y_test, original_probabilities
    )

    augmented_auc = roc_auc_score(
        y_test, augmented_probabilities
    )

    results.append({
        "Season": int(season),
        "Model": "Original Logistic Regression",
        "Accuracy": original_accuracy,
        "ROC AUC": original_auc
    })

    results.append({
        "Season": int(season),
        "Model": "Augmented Logistic Regression",
        "Accuracy": augmented_accuracy,
        "ROC AUC": augmented_auc
    })

    print(
        f"Original: Accuracy={original_accuracy:.4f}, "
        f"ROC AUC={original_auc:.4f}"
    )

    print(
        f"Augmented: Accuracy={augmented_accuracy:.4f}, "
        f"ROC AUC={augmented_auc:.4f}"
    )


# Save detailed results
results_df = pd.DataFrame(results)

results_file = MODEL_DIR / "team_swap_results.csv"
results_df.to_csv(results_file, index=False)


# Calculate average results
summary = results_df.groupby("Model").agg({
    "Accuracy": "mean",
    "ROC AUC": "mean"
}).reset_index()

summary = summary.round(4)

summary_file = MODEL_DIR / "team_swap_summary.csv"
summary.to_csv(summary_file, index=False)


print("\n" + "=" * 55)
print("TEAM-SWAPPING EXPERIMENT SUMMARY")
print("=" * 55)

print(summary.to_string(index=False))

print("\nResults saved:")
print(results_file)
print(summary_file)

print("\nExperiment completed successfully!")

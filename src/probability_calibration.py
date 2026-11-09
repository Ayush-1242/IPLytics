import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
print("Loading feature dataset...")

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# Target and features
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

X = df[FEATURES]
y = df[TARGET]


# Define models
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# Walk-forward validation
print("\nStarting walk-forward validation...")

seasons = sorted(df["season"].dropna().unique())

test_seasons = [season for season in seasons if season >= 2019]

all_results = []

calibration_results = []

for test_season in test_seasons:

    train_mask = df["season"] < test_season
    test_mask = df["season"] == test_season

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    print("\n" + "=" * 60)
    print("Training seasons: Before", int(test_season))
    print("Testing season:", int(test_season))
    print("Training matches:", len(X_train))
    print("Testing matches:", len(X_test))

    # Create fresh preprocessing for each model and fold
    def create_preprocessor():

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

     # Create a fresh Logistic Regression pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Generate predictions and probabilities
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Calculate calibration curve using 10 probability bins
    actual_rates, predicted_rates = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    for actual, predicted in zip(actual_rates, predicted_rates):
        calibration_results.append({
            "Season": int(test_season),
            "Actual Win Rate": actual,
            "Mean Predicted Probability": predicted
    })

    # Classification metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = (
        roc_auc_score(y_test, probabilities)
        if y_test.nunique() == 2
        else float("nan")
    )

    # Probability calibration metrics
    brier = brier_score_loss(y_test, probabilities)
    loss = log_loss(y_test, probabilities, labels=[0, 1])

    all_results.append({
        "Season": int(test_season),
        "Training Matches": len(X_train),
        "Testing Matches": len(X_test),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Brier Score": brier,
        "Log Loss": loss
    })

    print(
        f"Logistic Regression: "
        f"Accuracy={accuracy:.4f}, "
        f"ROC AUC={auc:.4f}, "
        f"Brier Score={brier:.4f}, "
        f"Log Loss={loss:.4f}"
    )


# Save season-wise calibration results
results_df = pd.DataFrame(all_results)

results_file = MODEL_DIR / "probability_calibration_results.csv"
results_df.to_csv(results_file, index=False)


# Calculate average performance across seasons
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC",
    "Brier Score",
    "Log Loss"
]

summary = results_df[metrics].mean().to_frame(
    name="Average Across Seasons"
).T.round(4)

summary_file = MODEL_DIR / "probability_calibration_summary.csv"
summary.to_csv(summary_file, index=False)


# Save calibration curve data
calibration_df = pd.DataFrame(calibration_results)

calibration_file = MODEL_DIR / "calibration_curve_data.csv"
calibration_df.to_csv(calibration_file, index=False)


# Display results
print("\n" + "=" * 60)
print("PROBABILITY CALIBRATION SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\nSEASON-WISE RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nResults saved:")
print(results_file)
print(summary_file)
print(calibration_file)

print("\nProbability calibration evaluation completed!")
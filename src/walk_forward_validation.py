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
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
models = {
    "Dummy Baseline": DummyClassifier(
        strategy="prior"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )
}


# Walk-forward validation
print("\nStarting walk-forward validation...")

seasons = sorted(df["season"].dropna().unique())

test_seasons = [season for season in seasons if season >= 2019]

all_results = []

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

    # Create fresh preprocessing for each fold
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

    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

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

        all_results.append({
            "Season": int(test_season),
            "Model": name,
            "Training Matches": len(X_train),
            "Testing Matches": len(X_test),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": auc
        })

        print(
            f"{name}: "
            f"Accuracy={accuracy:.4f}, "
            f"ROC AUC={auc:.4f}"
        )


# Save individual season results
results_df = pd.DataFrame(all_results)

results_file = MODEL_DIR / "walk_forward_results.csv"

results_df.to_csv(results_file, index=False)


# Calculate average performance
summary = results_df.groupby("Model").agg({
    "Accuracy": "mean",
    "Precision": "mean",
    "Recall": "mean",
    "F1 Score": "mean",
    "ROC AUC": "mean"
}).reset_index()

summary = summary.round(4)

summary_file = MODEL_DIR / "walk_forward_summary.csv"

summary.to_csv(summary_file, index=False)


# Display final results
print("\n" + "=" * 60)
print("WALK-FORWARD VALIDATION SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\nResults saved:")
print(results_file)
print(summary_file)

print("\nWalk-forward validation completed!")

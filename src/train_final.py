import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "processed" / "features.csv"
VALIDATION_FILE = ROOT_DIR / "models" / "algorithm_comparison_results.csv"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load historical data
print("Loading IPL dataset...")

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "team1_win"

categorical_features = [
    "team1", "team2", "venue", "city"
]

numeric_features = [
    "season",
    "team1_win_pct", "team2_win_pct",
    "team1_matches_played", "team2_matches_played",
    "team1_last5_win_pct", "team2_last5_win_pct",
    "h2h_team1_win_pct", "h2h_matches",
    "team1_venue_win_pct", "team2_venue_win_pct",
    "team1_venue_matches", "team2_venue_matches",
    "win_pct_difference", "last5_win_pct_difference",
    "venue_win_pct_difference", "h2h_advantage",
    "matches_played_difference", "venue_matches_difference",
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
    "team1_elo", "team2_elo", "elo_difference",
    "team1_last10_win_pct", "team2_last10_win_pct",
    "last10_win_pct_difference"
]

features = categorical_features + numeric_features

X = df[features]
y = df[TARGET]


# Define models
models = {
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


def create_pipeline(model):

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
        ("model", model)
    ])


# Select best algorithm using recent walk-forward results
print("\nSelecting the best algorithm...")

validation = pd.read_csv(VALIDATION_FILE)

latest_seasons = sorted(validation["Season"].unique())[-3:]

recent_results = validation[
    validation["Season"].isin(latest_seasons)
]

model_scores = (
    recent_results.groupby("Model")["ROC AUC"]
    .mean()
    .dropna()
    .sort_values(ascending=False)
)

if model_scores.empty:
    raise ValueError("No valid ROC AUC results found.")

best_model_name = model_scores.index[0]

print("\nRecent seasons used:", latest_seasons)
print("\nRecent model performance:")
print(model_scores.to_string())

print("\nSelected model:", best_model_name)


# Train and save all three models
print("\nTraining all models on historical data...")

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = create_pipeline(model)
    pipeline.fit(X, y)

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    model_path = MODEL_DIR / f"{safe_name}_final.joblib"

    joblib.dump(pipeline, model_path)

    print("Saved:", model_path)


# Save model selection information
selection_info = {
    "selected_model": best_model_name,
    "selection_metric": "Mean ROC AUC",
    "selection_method": "Most recent three walk-forward seasons",
    "seasons_used": [int(s) for s in latest_seasons],
    "model_scores": {
        name: round(float(score), 4)
        for name, score in model_scores.items()
    },
    "training_matches": len(df),
    "training_data_end": str(df["date"].max().date()),
    "prediction_timing": "Before toss"
}

with open(MODEL_DIR / "model_selection.json", "w") as file:
    json.dump(selection_info, file, indent=4)

print("\n" + "=" * 55)
print("IPLytics FINAL TRAINING COMPLETED")
print("=" * 55)
print("Selected model:", best_model_name)
print("Training matches:", len(df))
print("All three models saved successfully!")

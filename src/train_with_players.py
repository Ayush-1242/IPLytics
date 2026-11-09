import json
import joblib
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


# Define target and input features
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


# Chronological train-test split
print("Splitting data chronologically...")

split_index = int(len(df) * 0.80)
split_date = df.iloc[split_index]["date"]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))
print("Test period starts:", split_date.date())


# Preprocessing
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


# Train and evaluate
results = []
trained_models = {}

print("\nTraining models...\n")

for name, model in models.items():

    print("=" * 50)
    print("Training:", name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Season-wise evaluation
    test_results = df.loc[test_mask, ["season"]].copy()
    test_results["actual"] = y_test.values
    test_results["predicted"] = predictions

    print("\nSeason-wise Accuracy:")

    for season, group in test_results.groupby("season"):
        season_accuracy = accuracy_score(
            group["actual"],
            group["predicted"]
        )

        print(
            f"{int(season)}: "
            f"{season_accuracy:.4f} "
            f"({len(group)} matches)"
        ) 

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
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(auc, 4)
    })

    trained_models[name] = pipeline

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))


# Compare models
results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print(results_df.to_string(index=False))


# Select best model based on ROC AUC
best_model_name = results_df.sort_values(
    by="ROC AUC",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)


# Save best model and metrics
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)

with open(MODEL_DIR / "best_model_info.json", "w") as file:
    json.dump({
        "best_model": best_model_name,
        "selection_metric": "ROC AUC",
        "test_start_date": str(split_date.date())
    }, file, indent=4)

print("\nTraining completed successfully!")
print("Best model saved in:", MODEL_DIR)

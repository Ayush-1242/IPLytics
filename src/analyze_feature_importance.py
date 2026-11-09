import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "features.csv"
OUTPUT_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)


# Load dataset
print("Loading feature dataset...")

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["date", "match_id"]).reset_index(drop=True)


# Target
TARGET = "team1_win"


# Categorical features
categorical_features = [
    "team1",
    "team2",
    "venue",
    "city"
]


# Numeric features
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

    # Player features
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

    # Elo ratings
    "team1_elo",
    "team2_elo",
    "elo_difference",

    # Recent form
    "team1_last10_win_pct",
    "team2_last10_win_pct",
    "last10_win_pct_difference"
]


FEATURES = categorical_features + numeric_features

# Check required columns
missing_columns = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# Chronological 80/20 split
split_index = int(len(df) * 0.80)

split_date = df.iloc[split_index]["date"]

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


print("\nDataset split:")
print("Training matches:", len(train_df))
print("Testing matches:", len(test_df))
print("Test starting date:", split_date.date())


# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Logistic Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])


# Train model
print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\nMODEL PERFORMANCE")
print("------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC:  {auc:.4f}")


# Extract model coefficients
classifier = model.named_steps["classifier"]

feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

coefficients = classifier.coef_[0]

coefficient_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_importance": np.abs(coefficients)
})

coefficient_df = coefficient_df.sort_values(
    "absolute_importance",
    ascending=False
)

coefficient_df.to_csv(
    OUTPUT_DIR / "logistic_regression_coefficients.csv",
    index=False
)


print("\nTOP 15 POSITIVE COEFFICIENTS")
print("------------------------------")

positive = coefficient_df.sort_values(
    "coefficient",
    ascending=False
).head(15)

print(
    positive[["feature", "coefficient"]].to_string(index=False)
)


print("\nTOP 15 NEGATIVE COEFFICIENTS")
print("------------------------------")

negative = coefficient_df.sort_values(
    "coefficient"
).head(15)

print(
    negative[["feature", "coefficient"]].to_string(index=False)
)


# Permutation importance on untouched test data
print("\nCalculating permutation importance...")

permutation = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": permutation.importances_mean,
    "importance_std": permutation.importances_std
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "permutation_importance.csv",
    index=False
)


print("\nTOP 20 FEATURES BY PERMUTATION IMPORTANCE")
print("------------------------------------------")

print(
    importance_df.head(20).round(5).to_string(index=False)
)


print("\nBOTTOM 10 FEATURES BY PERMUTATION IMPORTANCE")
print("---------------------------------------------")

print(
    importance_df.tail(10).round(5).to_string(index=False)
)


print("\nAnalysis completed successfully!")

print("\nSaved files:")
print(OUTPUT_DIR / "logistic_regression_coefficients.csv")
print(OUTPUT_DIR / "permutation_importance.csv")

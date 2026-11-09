import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "features.csv"
OUTPUT_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)

BOOTSTRAP_REPEATS = 2000
RANDOM_STATE = 42

TARGET = "team1_win"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading feature dataset...")

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)


# --------------------------------------------------
# 3. DEFINE FEATURES
# --------------------------------------------------

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

    # Elo features
    "team1_elo",
    "team2_elo",
    "elo_difference",

    # Last 10 matches
    "team1_last10_win_pct",
    "team2_last10_win_pct",
    "last10_win_pct_difference"
]


h2h_features = [
    "h2h_team1_win_pct",
    "h2h_matches",
    "h2h_advantage"
]

team_identity_features = [
    "team1",
    "team2"
]


# --------------------------------------------------
# 4. MODEL CONFIGURATIONS
# --------------------------------------------------

configurations = {
    "Original": [],

    "Without Head-to-Head": h2h_features,

    "Without Team Identity": team_identity_features
}


# --------------------------------------------------
# 5. MODEL TRAINING FUNCTION
# --------------------------------------------------

def train_model(train_df, test_df, removed_features):

    current_categorical = [
        f for f in categorical_features
        if f not in removed_features
    ]

    current_numeric = [
        f for f in numeric_features
        if f not in removed_features
    ]

    current_features = (
        current_categorical + current_numeric
    )

    transformers = []

    if current_numeric:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("num", numeric_transformer, current_numeric)
        )

    if current_categorical:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("cat", categorical_transformer, current_categorical)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    X_train = train_df[current_features]
    y_train = train_df[TARGET]

    X_test = test_df[current_features]
    y_test = test_df[TARGET]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return y_test.to_numpy(), predictions, probabilities


# --------------------------------------------------
# 6. WALK-FORWARD MODEL EVALUATION
# --------------------------------------------------

print("\nTraining models for seasons 2019–2026...")

season_predictions = {}
season_results = []

for season in range(2019, 2027):

    train_df = df[
        df["date"].dt.year < season
    ].copy()

    test_df = df[
        df["date"].dt.year == season
    ].copy()

    if train_df.empty or test_df.empty:
        continue

    print(f"\nTraining and testing season {season}...")

    season_predictions[season] = {}

    for config_name, removed_features in configurations.items():

        y, predictions, probabilities = train_model(
            train_df,
            test_df,
            removed_features
        )

        accuracy = accuracy_score(y, predictions)
        auc = roc_auc_score(y, probabilities)

        season_predictions[season][config_name] = {
            "y": y,
            "predictions": predictions,
            "probabilities": probabilities
        }

        season_results.append({
            "season": season,
            "configuration": config_name,
            "accuracy": accuracy,
            "roc_auc": auc
        })

        print(
            f"{config_name}: "
            f"Accuracy={accuracy:.4f}, "
            f"ROC AUC={auc:.4f}"
        )


# --------------------------------------------------
# 7. ORIGINAL SEASON-WISE SUMMARY
# --------------------------------------------------

season_results_df = pd.DataFrame(season_results)

season_results_df.to_csv(
    OUTPUT_DIR / "bootstrap_season_results.csv",
    index=False
)

point_summary = (
    season_results_df
    .groupby("configuration")
    .agg(
        accuracy=("accuracy", "mean"),
        roc_auc=("roc_auc", "mean")
    )
)

print("\n\nORIGINAL WALK-FORWARD SUMMARY")
print("=" * 55)
print(point_summary.round(4).to_string())


# --------------------------------------------------
# 8. PAIRED STRATIFIED BOOTSTRAP
# --------------------------------------------------

print("\nStarting paired bootstrap analysis...")

rng = np.random.default_rng(RANDOM_STATE)

seasons = list(season_predictions.keys())

comparisons = [
    ("Without Head-to-Head", "Original"),
    ("Without Team Identity", "Original")
]

bootstrap_results = []

for candidate, baseline in comparisons:

    accuracy_differences = []
    auc_differences = []

    print(f"\nComparing {candidate} against {baseline}...")

    for repeat in range(BOOTSTRAP_REPEATS):

        baseline_accuracies = []
        candidate_accuracies = []

        baseline_aucs = []
        candidate_aucs = []

        for season in seasons:

            data = season_predictions[season]

            y = data[baseline]["y"]

            # Stratified resampling preserves
            # the original class counts per season.
            class_0 = np.where(y == 0)[0]
            class_1 = np.where(y == 1)[0]

            sampled_0 = rng.choice(
                class_0,
                size=len(class_0),
                replace=True
            )

            sampled_1 = rng.choice(
                class_1,
                size=len(class_1),
                replace=True
            )

            sampled_indices = np.concatenate([
                sampled_0,
                sampled_1
            ])

            rng.shuffle(sampled_indices)

            y_sample = y[sampled_indices]

            baseline_data = data[baseline]
            candidate_data = data[candidate]

            baseline_pred = (
                baseline_data["predictions"][sampled_indices]
            )

            candidate_pred = (
                candidate_data["predictions"][sampled_indices]
            )

            baseline_prob = (
                baseline_data["probabilities"][sampled_indices]
            )

            candidate_prob = (
                candidate_data["probabilities"][sampled_indices]
            )

            baseline_accuracies.append(
                accuracy_score(y_sample, baseline_pred)
            )

            candidate_accuracies.append(
                accuracy_score(y_sample, candidate_pred)
            )

            baseline_aucs.append(
                roc_auc_score(y_sample, baseline_prob)
            )

            candidate_aucs.append(
                roc_auc_score(y_sample, candidate_prob)
            )

        # Average season-level metrics equally,
        # matching the original walk-forward summary.
        accuracy_differences.append(
            np.mean(candidate_accuracies)
            - np.mean(baseline_accuracies)
        )

        auc_differences.append(
            np.mean(candidate_aucs)
            - np.mean(baseline_aucs)
        )

        if (repeat + 1) % 500 == 0:
            print(
                f"Completed {repeat + 1}/"
                f"{BOOTSTRAP_REPEATS} bootstrap samples"
            )

    # Point estimates from original predictions
    candidate_accuracy = point_summary.loc[
        candidate, "accuracy"
    ]

    baseline_accuracy = point_summary.loc[
        baseline, "accuracy"
    ]

    candidate_auc = point_summary.loc[
        candidate, "roc_auc"
    ]

    baseline_auc = point_summary.loc[
        baseline, "roc_auc"
    ]

    accuracy_difference = (
        candidate_accuracy - baseline_accuracy
    )

    auc_difference = (
        candidate_auc - baseline_auc
    )

    accuracy_ci = np.percentile(
        accuracy_differences,
        [2.5, 97.5]
    )

    auc_ci = np.percentile(
        auc_differences,
        [2.5, 97.5]
    )

    bootstrap_results.append({
        "comparison": f"{candidate} vs {baseline}",
        "accuracy_difference": accuracy_difference,
        "accuracy_ci_lower": accuracy_ci[0],
        "accuracy_ci_upper": accuracy_ci[1],
        "auc_difference": auc_difference,
        "auc_ci_lower": auc_ci[0],
        "auc_ci_upper": auc_ci[1]
    })


# --------------------------------------------------
# 9. DISPLAY AND SAVE RESULTS
# --------------------------------------------------

bootstrap_df = pd.DataFrame(bootstrap_results)

bootstrap_file = (
    OUTPUT_DIR / "paired_bootstrap_summary.csv"
)

bootstrap_df.to_csv(
    bootstrap_file,
    index=False
)

print("\n\nPAIRED BOOTSTRAP RESULTS")
print("=" * 75)

print(
    bootstrap_df.round(4).to_string(index=False)
)

print("\nInterpretation:")
print(
    "A confidence interval entirely above zero "
    "indicates evidence of improvement."
)

print(
    "A confidence interval entirely below zero "
    "indicates evidence of worse performance."
)

print(
    "An interval containing zero means the experiment "
    "does not establish a clear difference."
)

print("\nSaved files:")
print(
    OUTPUT_DIR / "bootstrap_season_results.csv"
)
print(bootstrap_file)

print("\nPaired bootstrap validation completed!")

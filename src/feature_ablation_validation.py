import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "features.csv"
OUTPUT_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("Loading feature dataset...")

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)


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


# --------------------------------------------------
# 2. DEFINE FEATURE GROUPS
# --------------------------------------------------

player_features = [
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
    "squad_bowling_strike_rate_difference"
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

venue_features = [
    "venue",
    "city"
]


# --------------------------------------------------
# 3. DEFINE EXPERIMENT CONFIGURATIONS
# --------------------------------------------------

configurations = {
    "All Features": [],

    "Without Player Features": player_features,

    "Without Team Identity": team_identity_features,

    "Without Venue and City": venue_features,

    "Without Head-to-Head": h2h_features,

    "Without Player and Venue": (
        player_features + venue_features
    )
}


# --------------------------------------------------
# 4. WALK-FORWARD VALIDATION
# --------------------------------------------------

test_seasons = range(2019, 2027)

results = []

print("\nStarting feature ablation validation...")


for season in test_seasons:

    print("\n" + "=" * 60)
    print(f"Testing season: {season}")
    print("=" * 60)

    train_df = df[
        df["date"].dt.year < season
    ].copy()

    test_df = df[
        df["date"].dt.year == season
    ].copy()

    if train_df.empty or test_df.empty:
        print("Skipping season due to missing data.")
        continue

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    print(
        f"Training matches: {len(train_df)} | "
        f"Testing matches: {len(test_df)}"
    )

    for config_name, removed_features in configurations.items():

        current_categorical = [
            feature
            for feature in categorical_features
            if feature not in removed_features
        ]

        current_numeric = [
            feature
            for feature in numeric_features
            if feature not in removed_features
        ]

        current_features = (
            current_categorical + current_numeric
        )

        X_train = train_df[current_features]
        X_test = test_df[current_features]

        transformers = []

        if current_numeric:
            numeric_transformer = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ])

            transformers.append(
                ("num", numeric_transformer, current_numeric)
            )

        if current_categorical:
            categorical_transformer = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ])

            transformers.append(
                (
                    "cat",
                    categorical_transformer,
                    current_categorical
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "season": season,
            "configuration": config_name,
            "train_matches": len(train_df),
            "test_matches": len(test_df),
            "features_used": len(current_features),
            "accuracy": accuracy,
            "roc_auc": auc
        })

        print(
            f"{config_name}: "
            f"Accuracy={accuracy:.4f}, "
            f"ROC AUC={auc:.4f}"
        )


# --------------------------------------------------
# 5. SAVE DETAILED RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_file = OUTPUT_DIR / "feature_ablation_results.csv"

results_df.to_csv(
    results_file,
    index=False
)


# --------------------------------------------------
# 6. GENERATE SUMMARY
# --------------------------------------------------

summary_df = (
    results_df
    .groupby("configuration")
    .agg(
        average_accuracy=("accuracy", "mean"),
        average_roc_auc=("roc_auc", "mean"),
        accuracy_std=("accuracy", "std"),
        roc_auc_std=("roc_auc", "std"),
        average_features=("features_used", "mean")
    )
    .reset_index()
    .sort_values(
        "average_roc_auc",
        ascending=False
    )
)

summary_file = OUTPUT_DIR / "feature_ablation_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n\n" + "=" * 70)
print("FEATURE ABLATION EXPERIMENT SUMMARY")
print("=" * 70)

print(
    summary_df.round(4).to_string(index=False)
)

print("\nDetailed results saved:")
print(results_file)

print("\nSummary saved:")
print(summary_file)

print("\nFeature ablation experiment completed successfully!")

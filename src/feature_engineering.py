from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


# Standardize historical IPL team names
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant"
}

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "matches.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "features.csv"


def create_features():

    print("Loading match dataset...")

    matches = pd.read_csv(
        INPUT_FILE,
        dtype={"match_id": str}
    )

    # Normalize team names
    for column in ["team1", "team2", "winner", "toss_winner"]:
        matches[column] = matches[column].replace(TEAM_ALIASES)

    matches["date"] = pd.to_datetime(matches["date"])
    matches["city"] = matches["city"].fillna("Unknown")
    matches["season"] = pd.to_numeric(matches["season"])

    matches = matches.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    # Historical team statistics
    team_stats = defaultdict(
        lambda: {
            "matches": 0,
            "wins": 0,
            "last5": deque(maxlen=5),
            "last10": deque(maxlen=10)
        }
    )

    # Team performance at each venue
    venue_stats = defaultdict(
        lambda: {
            "matches": 0,
            "wins": 0
        }
    )

    # Head-to-head statistics
    h2h_stats = defaultdict(
        lambda: {
            "matches": 0,
            "wins": defaultdict(int)
        }
    )

    # Elo rating system
    elo_ratings = defaultdict(lambda: 1500.0)

    ELO_K_FACTOR = 20
    ELO_SCALE = 400

    feature_records = []

    print("Generating historical features...")

    for match_date, daily_matches in matches.groupby("date", sort=True):

        daily_updates = []

        for _, match in daily_matches.iterrows():

            team1 = match["team1"]
            team2 = match["team2"]
            venue = match["venue"]

            # Current Elo ratings (before this match date)
            team1_elo = elo_ratings[team1]
            team2_elo = elo_ratings[team2]

# Expected probability of team 1 winning
            expected_team1 = 1 / (
                1 + 10 ** ((team2_elo - team1_elo) / ELO_SCALE)
            )

            team1_history = team_stats[team1]
            team2_history = team_stats[team2]

            venue1 = venue_stats[(team1, venue)]
            venue2 = venue_stats[(team2, venue)]

            pair = tuple(sorted([team1, team2]))
            h2h = h2h_stats[pair]

            toss_decision = str(
                match["toss_decision"]
            ).lower()

            # Historical win percentages
            team1_win_pct = (
                team1_history["wins"] / team1_history["matches"]
                if team1_history["matches"] else 0.5
            )

            team2_win_pct = (
                team2_history["wins"] / team2_history["matches"]
                if team2_history["matches"] else 0.5
            )

            # Recent form
            team1_last5 = (
                sum(team1_history["last5"]) /
                len(team1_history["last5"])
                if team1_history["last5"] else 0.5
            )

            team2_last5 = (
                sum(team2_history["last5"]) /
                len(team2_history["last5"])
                if team2_history["last5"] else 0.5
            )

            # Last 10 matches form
            team1_last10 = (
                sum(team1_history["last10"]) /
                len(team1_history["last10"])
                if team1_history["last10"] else 0.5
            )

            team2_last10 = (
                sum(team2_history["last10"]) /
                len(team2_history["last10"])
                if team2_history["last10"] else 0.5
)

            # Venue performance
            team1_venue_pct = (
                venue1["wins"] / venue1["matches"]
                if venue1["matches"] else 0.5
            )

            team2_venue_pct = (
                venue2["wins"] / venue2["matches"]
                if venue2["matches"] else 0.5
            )

            # Head-to-head performance
            h2h_team1_pct = (
                h2h["wins"][team1] / h2h["matches"]
                if h2h["matches"] else 0.5
            )

            # Toss information
            toss_winner_is_team1 = int(
                match["toss_winner"] == team1
            )

            toss_decision_bat = int(
                toss_decision == "bat"
            )

            toss_decision_field = int(
                toss_decision == "field"
            )

            # Feature dictionary
            features = {

                "match_id": str(match["match_id"]),

                # Overall historical performance
                "team1_win_pct": team1_win_pct,
                "team2_win_pct": team2_win_pct,

                "team1_matches_played": team1_history["matches"],
                "team2_matches_played": team2_history["matches"],

                # Recent form
                "team1_last5_win_pct": team1_last5,
                "team2_last5_win_pct": team2_last5,

                # Head-to-head
                "h2h_team1_win_pct": h2h_team1_pct,
                "h2h_matches": h2h["matches"],

                # Venue performance
                "team1_venue_win_pct": team1_venue_pct,
                "team2_venue_win_pct": team2_venue_pct,

                "team1_venue_matches": venue1["matches"],
                "team2_venue_matches": venue2["matches"],

                # Toss information
                "toss_winner_is_team1": toss_winner_is_team1,
                "toss_decision_bat": toss_decision_bat,
                "toss_decision_field": toss_decision_field,

                # NEW: Relative performance features
                "win_pct_difference": (
                    team1_win_pct - team2_win_pct
                ),

                "last5_win_pct_difference": (
                    team1_last5 - team2_last5
                ),

                "venue_win_pct_difference": (
                    team1_venue_pct - team2_venue_pct
                ),

                "team1_last10_win_pct": team1_last10,
                "team2_last10_win_pct": team2_last10,

                "last10_win_pct_difference": (
                    team1_last10 - team2_last10
                ),

                "h2h_advantage": (
                    h2h_team1_pct - 0.5
                ),

                "matches_played_difference": (
                    team1_history["matches"] -
                    team2_history["matches"]
                ),

                "venue_matches_difference": (
                    venue1["matches"] -
                    venue2["matches"]
                ),
                # Elo rating features
                "team1_elo": team1_elo,
                "team2_elo": team2_elo,
                "elo_difference": team1_elo - team2_elo,
            }

            feature_records.append(features)

            # Store match outcome for historical updates
            team1_won = int(match["team1_win"])

            # Calculate Elo rating changes
            elo_change = ELO_K_FACTOR * (
                team1_won - expected_team1
            )

            daily_updates.append(
                (
                    team1,
                    team2,
                    venue,
                    team1_won,
                    elo_change
                )
            )

        # Update histories after processing the entire date
        for team1, team2, venue, team1_won, elo_change in daily_updates:

            # Update Elo ratings after the match date
            elo_ratings[team1] += elo_change
            elo_ratings[team2] -= elo_change

            team2_won = 1 - team1_won

            team_stats[team1]["matches"] += 1
            team_stats[team1]["wins"] += team1_won
            team_stats[team1]["last5"].append(team1_won)
            team_stats[team1]["last10"].append(team1_won)

            team_stats[team2]["matches"] += 1
            team_stats[team2]["wins"] += team2_won
            team_stats[team2]["last5"].append(team2_won)
            team_stats[team2]["last10"].append(team2_won)

            venue_stats[(team1, venue)]["matches"] += 1
            venue_stats[(team1, venue)]["wins"] += team1_won

            venue_stats[(team2, venue)]["matches"] += 1
            venue_stats[(team2, venue)]["wins"] += team2_won

            pair = tuple(sorted([team1, team2]))

            h2h_stats[pair]["matches"] += 1
            h2h_stats[pair]["wins"][team1] += team1_won
            h2h_stats[pair]["wins"][team2] += team2_won

    # Combine original match information and engineered features
    features_df = pd.DataFrame(feature_records)

    features_df["match_id"] = features_df["match_id"].astype(str)

    final_df = matches.merge(
        features_df,
        on="match_id",
        how="inner",
        validate="one_to_one"
    )

    # Merge player-based squad features
    PLAYER_FEATURES_FILE = (
        BASE_DIR / "data" / "processed" / "player_features.csv"
    )

    player_features = pd.read_csv(
        PLAYER_FEATURES_FILE,
        dtype={"match_id": str}
    )

    final_df = final_df.merge(
        player_features,
        on="match_id",
        how="left",
        validate="one_to_one"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature engineering completed!")
    print("Total matches:", len(final_df))
    print("Total columns:", len(final_df.columns))
    print("Output saved to:", OUTPUT_FILE)

    print("\nGenerated features:")
    print(features_df.columns.tolist())


if __name__ == "__main__":
    create_features()

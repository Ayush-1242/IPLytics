from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

BALLS_FILE = BASE_DIR / "data" / "raw" / "all_matches.csv"
MATCHES_FILE = BASE_DIR / "data" / "processed" / "matches.csv"

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "player_features.csv"
)


# Standardize historical team names
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant"
}


# Wickets credited to the bowler
BOWLER_WICKETS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket"
}


STAT_COLUMNS = [
    "batting_runs",
    "balls_faced",
    "batting_outs",
    "runs_conceded",
    "legal_balls",
    "wickets"
]


def calculate_squad_features(players, career_stats):

    totals = {
        column: 0.0
        for column in STAT_COLUMNS
    }

    for player in players:
        stats = career_stats[player]

        for column in STAT_COLUMNS:
            totals[column] += stats[column]

    runs = totals["batting_runs"]
    outs = totals["batting_outs"]
    balls = totals["balls_faced"]

    conceded = totals["runs_conceded"]
    legal_balls = totals["legal_balls"]
    wickets = totals["wickets"]

    batting_average = (
        runs / outs if outs > 0 else float("nan")
    )

    batting_strike_rate = (
        100 * runs / balls if balls > 0 else float("nan")
    )

    bowling_economy = (
        6 * conceded / legal_balls
        if legal_balls > 0 else float("nan")
    )

    bowling_strike_rate = (
        legal_balls / wickets
        if wickets > 0 else float("nan")
    )

    return {
        "squad_batting_average": batting_average,
        "squad_batting_strike_rate": batting_strike_rate,
        "squad_bowling_economy": bowling_economy,
        "squad_bowling_strike_rate": bowling_strike_rate,
        "squad_player_count": len(players)
    }


def create_player_features():

    print("Loading match dataset...")

    matches = pd.read_csv(
        MATCHES_FILE,
        dtype={"match_id": "string"}
    )

    matches["date"] = pd.to_datetime(matches["date"])

    for column in ["team1", "team2"]:
        matches[column] = matches[column].replace(TEAM_ALIASES)

    matches = matches.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    print("Loading ball-by-ball dataset...")

    ball_columns = [
        "match_id",
        "batting_team",
        "bowling_team",
        "striker",
        "non_striker",
        "bowler",
        "runs_off_bat",
        "wides",
        "noballs",
        "wicket_type",
        "player_dismissed"
    ]

    balls = pd.read_csv(
        BALLS_FILE,
        usecols=ball_columns,
        dtype={"match_id": "string"},
        low_memory=False
    )

    for column in ["batting_team", "bowling_team"]:
        balls[column] = balls[column].replace(TEAM_ALIASES)

    for column in ["runs_off_bat", "wides", "noballs"]:
        balls[column] = (
            pd.to_numeric(balls[column], errors="coerce")
            .fillna(0)
        )

    print("Calculating player match statistics...")

    # --------------------------------
    # Batting statistics
    # --------------------------------

    batting = balls[
        [
            "match_id",
            "batting_team",
            "striker",
            "runs_off_bat",
            "wides",
            "player_dismissed"
        ]
    ].copy()

    batting = batting.rename(columns={
        "batting_team": "team",
        "striker": "player"
    })

    # A wide does not count as a ball faced.
    batting["is_ball_faced"] = (
        batting["wides"] == 0
    ).astype(int)

    batting["is_out"] = (
        batting["player_dismissed"].notna()
        & batting["player_dismissed"].eq(batting["player"])
    ).astype(int)

    batting_stats = (
        batting.dropna(subset=["player"])
        .groupby(
            ["match_id", "team", "player"],
            as_index=False
        )
        .agg(
            batting_runs=("runs_off_bat", "sum"),
            balls_faced=("is_ball_faced", "sum"),
            batting_outs=("is_out", "sum")
        )
    )

    # --------------------------------
    # Bowling statistics
    # --------------------------------

    bowling = balls[
        [
            "match_id",
            "bowling_team",
            "bowler",
            "runs_off_bat",
            "wides",
            "noballs",
            "wicket_type"
        ]
    ].copy()

    bowling = bowling.rename(columns={
        "bowling_team": "team",
        "bowler": "player"
    })

    bowling["legal_ball"] = (
        (bowling["wides"] == 0)
        & (bowling["noballs"] == 0)
    ).astype(int)

    # Runs charged to the bowler.
    # Byes and legbyes are not charged to the bowler.
    bowling["runs_conceded"] = (
        bowling["runs_off_bat"]
        + bowling["wides"]
        + bowling["noballs"]
    )

    bowling["wickets"] = (
        bowling["wicket_type"]
        .fillna("")
        .str.lower()
        .isin(BOWLER_WICKETS)
        .astype(int)
    )

    bowling_stats = (
        bowling.dropna(subset=["player"])
        .groupby(
            ["match_id", "team", "player"],
            as_index=False
        )
        .agg(
            runs_conceded=("runs_conceded", "sum"),
            legal_balls=("legal_ball", "sum"),
            wickets=("wickets", "sum")
        )
    )

    # --------------------------------
    # Identify players who appeared
    # --------------------------------

    striker_appearances = balls[
        ["match_id", "batting_team", "striker"]
    ].rename(columns={
        "batting_team": "team",
        "striker": "player"
    })

    non_striker_appearances = balls[
        ["match_id", "batting_team", "non_striker"]
    ].rename(columns={
        "batting_team": "team",
        "non_striker": "player"
    })

    bowler_appearances = balls[
        ["match_id", "bowling_team", "bowler"]
    ].rename(columns={
        "bowling_team": "team",
        "bowler": "player"
    })

    appearances = pd.concat(
        [
            striker_appearances,
            non_striker_appearances,
            bowler_appearances
        ],
        ignore_index=True
    )

    appearances = appearances.dropna(
        subset=["match_id", "team", "player"]
    ).drop_duplicates(
        subset=["match_id", "team", "player"]
    )

    # Combine batting and bowling statistics
    player_matches = appearances.merge(
        batting_stats,
        on=["match_id", "team", "player"],
        how="left"
    )

    player_matches = player_matches.merge(
        bowling_stats,
        on=["match_id", "team", "player"],
        how="left"
    )

    player_matches[STAT_COLUMNS] = (
        player_matches[STAT_COLUMNS].fillna(0)
    )

    # Match ID -> player statistics
    players_by_match = {}

    for match_id, group in player_matches.groupby("match_id"):

        players_by_match[str(match_id)] = (
            group.to_dict("records")
        )

    print("Generating pre-match squad features...")

    # Career statistics for each player
    career_stats = defaultdict(
        lambda: {
            column: 0.0
            for column in STAT_COLUMNS
        }
    )

    # Players from each team's previous five matches
    recent_team_players = defaultdict(
        lambda: deque(maxlen=5)
    )

    feature_records = []

    for match_date, daily_matches in matches.groupby(
        "date",
        sort=True
    ):

        daily_updates = []

        # Calculate features before updating any results
        for _, match in daily_matches.iterrows():

            match_id = str(match["match_id"])

            team1 = match["team1"]
            team2 = match["team2"]

            # Players from the previous five team matches
            team1_players = set().union(
                *recent_team_players[team1]
            ) if recent_team_players[team1] else set()

            team2_players = set().union(
                *recent_team_players[team2]
            ) if recent_team_players[team2] else set()

            team1_features = calculate_squad_features(
                team1_players,
                career_stats
            )

            team2_features = calculate_squad_features(
                team2_players,
                career_stats
            )

            features = {
                "match_id": match_id,

                "team1_squad_batting_average":
                    team1_features["squad_batting_average"],

                "team2_squad_batting_average":
                    team2_features["squad_batting_average"],

                "team1_squad_batting_strike_rate":
                    team1_features["squad_batting_strike_rate"],

                "team2_squad_batting_strike_rate":
                    team2_features["squad_batting_strike_rate"],

                "team1_squad_bowling_economy":
                    team1_features["squad_bowling_economy"],

                "team2_squad_bowling_economy":
                    team2_features["squad_bowling_economy"],

                "team1_squad_bowling_strike_rate":
                    team1_features["squad_bowling_strike_rate"],

                "team2_squad_bowling_strike_rate":
                    team2_features["squad_bowling_strike_rate"],

                "team1_squad_player_count":
                    team1_features["squad_player_count"],

                "team2_squad_player_count":
                    team2_features["squad_player_count"],

                "squad_batting_average_difference": (
                    team1_features["squad_batting_average"]
                    - team2_features["squad_batting_average"]
                ),

                "squad_batting_strike_rate_difference": (
                    team1_features["squad_batting_strike_rate"]
                    - team2_features["squad_batting_strike_rate"]
                ),

                "squad_bowling_economy_difference": (
                    team1_features["squad_bowling_economy"]
                    - team2_features["squad_bowling_economy"]
                ),

                "squad_bowling_strike_rate_difference": (
                    team1_features["squad_bowling_strike_rate"]
                    - team2_features["squad_bowling_strike_rate"]
                )
            }

            feature_records.append(features)

            match_players = players_by_match.get(
                match_id, []
            )

            daily_updates.append(
                (
                    team1,
                    team2,
                    match_players
                )
            )

        # Update player statistics only after the entire date
        for team1, team2, match_players in daily_updates:

            team_rosters = defaultdict(set)

            for player_record in match_players:

                team = player_record["team"]
                player = player_record["player"]

                team_rosters[team].add(player)

                for column in STAT_COLUMNS:
                    career_stats[player][column] += float(
                        player_record[column]
                    )

            # Store the latest match rosters
            recent_team_players[team1].append(
                team_rosters[team1]
            )

            recent_team_players[team2].append(
                team_rosters[team2]
            )

    # Save player features
    player_features = pd.DataFrame(feature_records)

    player_features["match_id"] = (
        player_features["match_id"].astype(str)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    player_features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nPlayer feature engineering completed!")
    print("Total matches:", len(player_features))
    print("Total columns:", len(player_features.columns))
    print("Saved to:", OUTPUT_FILE)
    print("\nGenerated features:")
    print(player_features.columns.tolist())


if __name__ == "__main__":
    create_player_features()
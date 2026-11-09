import csv
from pathlib  import Path
import pandas as pd
# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

BALLS_FILE = RAW_DIR / "all_matches.csv"
OUTPUT_FILE = PROCESSED_DIR / "matches.csv"


def read_match_info(file_path):
    """Extract match metadata from one Cricsheet info CSV."""

    info = {}

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 3 or row[0] != "info":
                continue

            key = row[1]

            if key in {
                "team",
                "winner",
                "toss_winner",
                "toss_decision",
                "date",
                "venue",
                "city",
                "season",
                "match_type",
                "event",
            }:
                info.setdefault(key, []).append(row[2])

    def first_value(key):
        values = info.get(key, [])
        return values[0] if values else None

    teams = info.get("team", [])
    winners = info.get("winner", [])

    return {
        "match_id": file_path.name.replace("_info.csv", ""),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "winner": first_value("winner"),
        "toss_winner": first_value("toss_winner"),
        "toss_decision": first_value("toss_decision"),
        "date": first_value("date"),
        "venue": first_value("venue"),
        "city": first_value("city"),
        "season": first_value("season"),
        "match_type": first_value("match_type"),
        "event": first_value("event"),
        "has_result": bool(winners),
    }


def create_match_dataset():
    print("Reading ball-by-ball dataset...")

    if not BALLS_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {BALLS_FILE}")

    balls = pd.read_csv(
    BALLS_FILE,
    usecols=["match_id"],
    dtype={"match_id": "string"}
    )

    print(f"Ball-by-ball rows: {len(balls):,}")
    print(f"Unique match IDs in ball data: {balls['match_id'].nunique():,}")

    print("\nReading match information files...")

    info_files = sorted(RAW_DIR.glob("*_info.csv"))

    if not info_files:
        raise FileNotFoundError("No *_info.csv files found in data/raw")

    match_records = [
        read_match_info(file_path)
        for file_path in info_files
    ]

    matches = pd.DataFrame(match_records)

    matches["match_id"] = matches["match_id"].astype(str)

    # Remove matches that do not have a normal winner.
    # Tied, abandoned and no-result matches are excluded from this
    # first binary classification dataset.
    valid_winner = (
    matches["winner"].eq(matches["team1"])
    | matches["winner"].eq(matches["team2"])
    )

    matches = matches[
        matches["winner"].notna()
        & matches["team1"].notna()
        & matches["team2"].notna()
        & valid_winner
    ].copy()

    # Keep only matches also present in the ball-by-ball dataset.
    ball_match_ids = set(balls["match_id"].astype(str).unique())

    matches = matches[
        matches["match_id"].isin(ball_match_ids)
    ].copy()

    # Create binary target.
    matches["team1_win"] = (
        matches["winner"] == matches["team1"]
    ).astype(int)

    matches["date"] = pd.to_datetime(
    matches["date"],
    errors="coerce"
    )

    matches["season"] = pd.to_numeric(
        matches["season"],
        errors="coerce"
    )

# Fill missing seasons using the match date.
    matches["season"] = matches["season"].fillna(
        matches["date"].dt.year
    )

    matches = matches.sort_values("date").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed!")
    print(f"Information files found: {len(info_files):,}")
    print(f"Matches in processed dataset: {len(matches):,}")
    print(f"Team 1 wins: {matches['team1_win'].sum():,}")
    print(f"Team 2 wins: {(matches['team1_win'] == 0).sum():,}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_match_dataset()
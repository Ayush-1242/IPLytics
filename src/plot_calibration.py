import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "models" / "calibration_curve_data.csv"
OUTPUT_FILE = ROOT_DIR / "models" / "calibration_curve.png"

# Load calibration data
df = pd.read_csv(DATA_FILE)

plt.figure(figsize=(9, 7))

# Plot each season
for season, group in df.groupby("Season"):
    group = group.sort_values("Mean Predicted Probability")

    plt.plot(
        group["Mean Predicted Probability"],
        group["Actual Win Rate"],
        marker="o",
        label=str(season)
    )

# Perfect calibration reference
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Win Probability")
plt.ylabel("Actual Win Rate")
plt.title("IPLytics - Logistic Regression Calibration")
plt.legend(title="Season")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Calibration graph saved to: {OUTPUT_FILE}")    
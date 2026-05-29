from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
from q1_load import start_client, load_data

# Set params. for filtering
OUTPUT_DIR = Path("q1_figs")
HIST_SAMPLE_FRAC = 0.01
DISTANCE_MAX = 40
FARE_MAX = 200
TIP_MAX = 100
PASSENGER_MAX = 6

# Function to prep data (remove outliers, add tip%, add bins, etc.)
def prep_data(df):
    cols = [
        "tpep_pickup_datetime",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]
    df = df[cols].copy()

    df["tpep_pickup_datetime"] = dd.to_datetime(
        df["tpep_pickup_datetime"], errors="coerce"
    )

    for col in ["passenger_count", "trip_distance", "fare_amount", "tip_amount", "total_amount"]:
        df[col] = dd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] <= DISTANCE_MAX)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] <= FARE_MAX)
        & (df["tip_amount"] >= 0)
        & (df["tip_amount"] <= TIP_MAX)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= PASSENGER_MAX)
        & (df["total_amount"] > 0)
    ]

    df["tip_pct"] = (df["tip_amount"] / df["fare_amount"]) * 100
    df = df[(df["tip_pct"] >= 0) & (df["tip_pct"] <= 100)]

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["distance_bin"] = df["trip_distance"].map_partitions(
        lambda s: pd.cut(
            s,
            bins=[0, 1, 2, 3, 5, 10, 20, 40],
            labels=["0-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-40"],
            include_lowest=True,
        )
    )
    return df

# Plot histograms for the distirbution of fare amounts and trip distances
def plot_histograms(df):
    sample = (
        df[["fare_amount", "trip_distance"]]
        .sample(frac=HIST_SAMPLE_FRAC, random_state=30123)
        .compute()
    )

    plt.figure(figsize=(8, 5))
    sns.histplot(sample["fare_amount"], bins=50)
    plt.title("Fare Amount Distribution")
    plt.xlabel("fare_amount")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fare_amount_hist.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(sample["trip_distance"], bins=50)
    plt.title("Trip Distance Distribution")
    plt.xlabel("trip_distance")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trip_distance_hist.png", dpi=300)
    plt.close()

# Plot trip distance bins and tip amounts
def plot_distance_vs_tip(df):
    summary = (
        df.groupby("distance_bin")
        .agg({"tip_amount": "mean"})
        .compute()
        .reset_index()
        .sort_values("distance_bin")
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="distance_bin", y="tip_amount")
    plt.title("Mean Tip Amount by Trip Distance Bin")
    plt.xlabel("trip_distance bin")
    plt.ylabel("mean tip_amount")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_bin_mean_tip.png", dpi=300)
    plt.close()

    print("\nMean tip amount by trip distance bin:")
    print(summary.to_string(index=False))

# Plot tip% by passenger count
def plot_passenger_vs_tip_pct(df):
    summary = (
        df.groupby("passenger_count")
        .agg({"tip_pct": "mean"})
        .compute()
        .reset_index()
        .sort_values("passenger_count")
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="passenger_count", y="tip_pct")
    plt.title("Mean Tip Percentage by Passenger Count")
    plt.xlabel("passenger_count")
    plt.ylabel("mean tip_pct")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "passenger_count_mean_tip_pct.png", dpi=300)
    plt.close()

    print("\nMean tip percentage by passenger count:")
    print(summary.to_string(index=False))

# Plot mean tip amount by time bins (of pickup)
def plot_hour_vs_tip(df):
    summary = (
        df.groupby("pickup_hour")
        .agg({"tip_amount": "mean"})
        .compute()
        .reset_index()
        .sort_values("pickup_hour")
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary, x="pickup_hour", y="tip_amount", marker="o")
    plt.title("Mean Tip Amount by Pickup Hour")
    plt.xlabel("pickup_hour")
    plt.ylabel("mean tip_amount")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pickup_hour_mean_tip.png", dpi=300)
    plt.close()

    print("\nMean tip amount by pickup hour:")
    print(summary.to_string(index=False))



def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    client = start_client()
    print(client)

    df = load_data()
    df = prep_data(df)

    print("\nColumns used:")
    print(df.columns)

    plot_histograms(df)
    plot_distance_vs_tip(df)
    plot_passenger_vs_tip_pct(df)
    plot_hour_vs_tip(df)

    print(f"\nSaved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
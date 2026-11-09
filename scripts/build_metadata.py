"""
Constructs the unified metadata.csv and storm-wise chronological train/validation/test splits.
Calculates multi-horizon forecasting targets (+6h, +12h, +24h) and intensity trends.
Ensures zero data leakage by splitting strictly by cyclone ID (SID) and season chronology.
"""

import os
import pandas as pd
import numpy as np

TRACKS_PATH = os.path.join("datasets", "processed", "storm_tracks.csv")
METADATA_PATH = os.path.join("datasets", "processed", "metadata.csv")
SPLITS_DIR = os.path.join("datasets", "splits")


def compute_multi_horizon_targets(group):
    """
    Computes +6h, +12h, +24h forecast targets for wind speed, pressure, and position.
    Also determines intensity trend (Strengthening, Weakening, Stable).
    """
    group = group.sort_values(by="ISO_TIME").copy()
    group["ISO_TIME"] = pd.to_datetime(group["ISO_TIME"])

    # Identify temporal steps (most observations are at 3-hour or 6-hour intervals)
    # We use time-based merge to find future observations
    for hours in [6, 12, 24]:
        group[f"TARGET_TIME_+{hours}h"] = group["ISO_TIME"] + pd.Timedelta(hours=hours)

    records = []
    times = group["ISO_TIME"].values
    winds = group["WIND_KTS"].values
    lats = group["LAT"].values
    lons = group["LON"].values
    press = group["PRES_HPA"].values

    for idx, row in group.iterrows():
        t = row["ISO_TIME"]
        rec = row.to_dict()

        # Multi-horizon lookups
        for h in [6, 12, 24]:
            target_t = t + pd.Timedelta(hours=h)
            # Find closest future observation within +/- 1.5 hours tolerance
            diffs = np.abs((group["ISO_TIME"] - target_t).dt.total_seconds() / 3600.0)
            closest_idx = diffs.idxmin()
            if diffs.loc[closest_idx] <= 2.0:
                rec[f"FUTURE_WIND_+{h}h"] = group.loc[closest_idx, "WIND_KTS"]
                rec[f"FUTURE_PRES_+{h}h"] = group.loc[closest_idx, "PRES_HPA"]
                rec[f"FUTURE_LAT_+{h}h"] = group.loc[closest_idx, "LAT"]
                rec[f"FUTURE_LON_+{h}h"] = group.loc[closest_idx, "LON"]
                rec[f"FUTURE_CLASS_+{h}h"] = group.loc[closest_idx, "IMD_CLASS_NAME"]
            else:
                rec[f"FUTURE_WIND_+{h}h"] = np.nan
                rec[f"FUTURE_PRES_+{h}h"] = np.nan
                rec[f"FUTURE_LAT_+{h}h"] = np.nan
                rec[f"FUTURE_LON_+{h}h"] = np.nan
                rec[f"FUTURE_CLASS_+{h}h"] = np.nan

        # Trend classification (+6h or +12h):
        # Strengthening (> +5 kt change), Weakening (< -5 kt change), Stable (-5 to +5 kt)
        future_w6 = rec.get("FUTURE_WIND_+6h")
        current_w = rec.get("WIND_KTS")
        if pd.notna(future_w6) and pd.notna(current_w):
            dw = future_w6 - current_w
            if dw > 5.0:
                rec["INTENSITY_TREND"] = "Strengthening"
            elif dw < -5.0:
                rec["INTENSITY_TREND"] = "Weakening"
            else:
                rec["INTENSITY_TREND"] = "Stable"
        else:
            rec["INTENSITY_TREND"] = np.nan

        # Image placeholder / association
        rec["IMAGE_PATH"] = f"datasets/processed/images/{rec['SID']}_{t.strftime('%Y%m%d_%H%M')}.png"

        records.append(rec)

    return pd.DataFrame(records)


def build_metadata():
    print(f"Loading storm tracks from {TRACKS_PATH}...")
    df = pd.read_csv(TRACKS_PATH)
    print(f"Loaded {len(df):,} track points.")

    # Filter records with wind speed
    df_valid = df.dropna(subset=["WIND_KTS"]).copy()

    # Process targets for each storm
    print("Computing multi-horizon forecasting targets (+6h, +12h, +24h) and intensity trends...")
    processed_groups = []
    for sid, group in df_valid.groupby("SID"):
        p_group = compute_multi_horizon_targets(group)
        processed_groups.append(p_group)

    df_meta = pd.concat(processed_groups, ignore_index=True)
    df_meta.to_csv(METADATA_PATH, index=False)
    print(f"Saved complete metadata to {METADATA_PATH} ({len(df_meta):,} records).")

    # Chronological storm-wise split:
    # Train: 1980 - 2019
    # Validation: 2020 - 2022
    # Test: 2023 - 2026 (Unseen recent holdout, matching Test 2025 presentation benchmark)
    train_df = df_meta[df_meta["SEASON"] <= 2019].copy()
    val_df = df_meta[(df_meta["SEASON"] >= 2020) & (df_meta["SEASON"] <= 2022)].copy()
    test_df = df_meta[df_meta["SEASON"] >= 2023].copy()

    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "validation.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n--- DATASET SPLITS SUMMARY ---")
    print(f"Train Set (1980 - 2019): {len(train_df):,} records, {train_df['SID'].nunique()} cyclones")
    print(f"Validation Set (2020 - 2022): {len(val_df):,} records, {val_df['SID'].nunique()} cyclones")
    print(f"Test Set (2023 - 2026 Holdout): {len(test_df):,} records, {test_df['SID'].nunique()} cyclones")
    print("\nTest Set Cyclones:", test_df["STORM_NAME"].unique()[:10])


if __name__ == "__main__":
    build_metadata()

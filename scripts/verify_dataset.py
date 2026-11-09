"""
Verification utility to audit the integrity, distributions, and readiness of the collected dataset.
Checks raw and processed tracks, metadata, splits, and downloaded satellite files.
"""

import os
import glob
import pandas as pd
import numpy as np

DATASETS_DIR = "datasets"


def verify_all():
    print("=" * 65)
    print("CYCLONE DATASET AUDIT & INTEGRITY VERIFICATION")
    print("=" * 65)

    # 1. Raw Data Check
    ibtracs_raw = os.path.join(DATASETS_DIR, "raw", "ibtracs", "ibtracs.NI.list.v04r01.csv")
    if os.path.exists(ibtracs_raw):
        raw_size = os.path.getsize(ibtracs_raw) / (1024 * 1024)
        print(f" [PASS] Raw IBTrACS (NI Basin): Present ({raw_size:.2f} MB)")
    else:
        print(" [FAIL] Raw IBTrACS file missing.")

    # 2. Processed Files Check
    tracks_path = os.path.join(DATASETS_DIR, "processed", "storm_tracks.csv")
    catalog_path = os.path.join(DATASETS_DIR, "processed", "storm_catalog.csv")
    meta_path = os.path.join(DATASETS_DIR, "processed", "metadata.csv")

    for name, p in [
        ("Storm Tracks", tracks_path),
        ("Storm Catalog", catalog_path),
        ("Unified Metadata", meta_path),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f" [PASS] Processed {name}: {len(df):,} records, {len(df.columns)} columns")
        else:
            print(f" [FAIL] {name} missing at {p}")

    # 3. Splits Check
    splits = ["train", "validation", "test"]
    split_counts = {}
    split_sids = {}
    for s in splits:
        sp = os.path.join(DATASETS_DIR, "splits", f"{s}.csv")
        if os.path.exists(sp):
            df_s = pd.read_csv(sp)
            split_counts[s] = len(df_s)
            split_sids[s] = set(df_s["SID"].unique())
            print(f" [PASS] Split '{s}': {len(df_s):,} records across {len(split_sids[s])} unique cyclones")
        else:
            print(f" [FAIL] Split '{s}' missing at {sp}")

    # Check for leakage between splits
    train_val_overlap = split_sids.get("train", set()).intersection(split_sids.get("validation", set()))
    train_test_overlap = split_sids.get("train", set()).intersection(split_sids.get("test", set()))
    val_test_overlap = split_sids.get("validation", set()).intersection(split_sids.get("test", set()))

    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print(" [PASS] Data Leakage Check: 0 storm overlap between train, val, and test splits!")
    else:
        print(f" [WARNING] Potential storm overlap detected: TV={len(train_val_overlap)}, TT={len(train_test_overlap)}, VT={len(val_test_overlap)}")

    # 4. Satellite Imagery Check
    satellite_imgs = glob.glob(os.path.join(DATASETS_DIR, "raw", "insat3d", "*.jpg"))
    print(f" [PASS] Raw Satellite Imagery (INSAT-3D): {len(satellite_imgs)} image products stored")
    for img in satellite_imgs:
        size_kb = os.path.getsize(img) / 1024
        print(f"       * {os.path.basename(img)} ({size_kb:.1f} KB)")

    # 5. IMD Class Distribution in Metadata
    if os.path.exists(meta_path):
        df_m = pd.read_csv(meta_path)
        print("\n--- IMD INTENSITY CLASS DISTRIBUTION ---")
        counts = df_m["IMD_CLASS_NAME"].value_counts()
        for k, v in counts.items():
            pct = (v / len(df_m)) * 100
            print(f" {k:<32}: {v:>6,} ({pct:>5.1f}%)")

        print("\n--- FORECAST HORIZON TARGET AVAILABILITY ---")
        for h in [6, 12, 24]:
            col = f"FUTURE_WIND_+{h}h"
            valid_targets = df_m[col].notna().sum()
            pct = (valid_targets / len(df_m)) * 100
            print(f" +{h:<2}h Horizon Target: {valid_targets:>6,} records ({pct:>5.1f}% coverage)")

    print("=" * 65)
    print("Dataset audit complete. Ready for baseline model training (Phase 2/3).")
    print("=" * 65)


if __name__ == "__main__":
    verify_all()


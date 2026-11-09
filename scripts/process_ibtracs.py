"""
Processes raw NOAA IBTrACS v4 dataset for North Indian Ocean (NI) basin.
Extracts Bay of Bengal (BB) and Arabian Sea (AS) storms.
Cleans coordinates, wind speed, pressure, and maps them to official IMD categories.
"""

import os
import pandas as pd
import numpy as np

RAW_IBTRACS_PATH = os.path.join("datasets", "raw", "ibtracs", "ibtracs.NI.list.v04r01.csv")
PROCESSED_TRACKS_PATH = os.path.join("datasets", "processed", "storm_tracks.csv")
STORM_CATALOG_PATH = os.path.join("datasets", "processed", "storm_catalog.csv")


def map_imd_intensity(wind_kt, raw_grade=""):
    """
    Maps wind speed (knots) and raw RSMC grade to official IMD cyclone categories:
    1: Depression (17 - 27 kt)
    2: Deep Depression (28 - 33 kt)
    3: Cyclonic Storm (34 - 47 kt)
    4: Severe Cyclonic Storm (48 - 63 kt)
    5: Very Severe Cyclonic Storm (64 - 89 kt)
    6: Extremely Severe Cyclonic Storm (90 - 119 kt)
    7: Super Cyclonic Storm (>= 120 kt)
    0: Low Pressure / Sub-depression (< 17 kt or unknown)
    """
    grade = str(raw_grade).strip().upper()
    grade_map = {
        "D": (1, "Depression"),
        "DD": (2, "Deep Depression"),
        "CS": (3, "Cyclonic Storm"),
        "SCS": (4, "Severe Cyclonic Storm"),
        "VSCS": (5, "Very Severe Cyclonic Storm"),
        "ESCS": (6, "Extremely Severe Cyclonic Storm"),
        "SUCS": (7, "Super Cyclonic Storm"),
        "SUPS": (7, "Super Cyclonic Storm"),
    }
    if grade in grade_map:
        return grade_map[grade]

    if pd.isna(wind_kt) or wind_kt < 17:
        return (0, "Low Pressure Area")
    elif 17 <= wind_kt <= 27:
        return (1, "Depression")
    elif 28 <= wind_kt <= 33:
        return (2, "Deep Depression")
    elif 34 <= wind_kt <= 47:
        return (3, "Cyclonic Storm")
    elif 48 <= wind_kt <= 63:
        return (4, "Severe Cyclonic Storm")
    elif 64 <= wind_kt <= 89:
        return (5, "Very Severe Cyclonic Storm")
    elif 90 <= wind_kt <= 119:
        return (6, "Extremely Severe Cyclonic Storm")
    else:
        return (7, "Super Cyclonic Storm")


def process_ibtracs():
    print(f"Loading raw IBTrACS data from {RAW_IBTRACS_PATH}...")
    # Row 1 (index 1) in IBTrACS contains unit headers; skip it.
    df = pd.read_csv(RAW_IBTRACS_PATH, skiprows=[1], low_memory=False)
    print(f"Loaded {len(df):,} total raw observations.")

    # Filter to North Indian Ocean basin (Subbasins: Bay of Bengal 'BB', Arabian Sea 'AS', or Basin 'NI')
    df_ni = df[df["BASIN"].isin(["NI"]) | df["SUBBASIN"].isin(["BB", "AS"])].copy()

    # Convert types
    df_ni["SEASON"] = pd.to_numeric(df_ni["SEASON"], errors="coerce")
    df_ni["LAT"] = pd.to_numeric(df_ni["LAT"], errors="coerce")
    df_ni["LON"] = pd.to_numeric(df_ni["LON"], errors="coerce")
    df_ni["ISO_TIME"] = pd.to_datetime(df_ni["ISO_TIME"], errors="coerce")

    # Agency wind speeds & pressure
    # Priority: RSMC New Delhi (IMD) -> WMO -> USA (JTWC)
    df_ni["NEWDELHI_WIND"] = pd.to_numeric(df_ni["NEWDELHI_WIND"], errors="coerce")
    df_ni["NEWDELHI_PRES"] = pd.to_numeric(df_ni["NEWDELHI_PRES"], errors="coerce")
    df_ni["WMO_WIND"] = pd.to_numeric(df_ni["WMO_WIND"], errors="coerce")
    df_ni["WMO_PRES"] = pd.to_numeric(df_ni["WMO_PRES"], errors="coerce")
    df_ni["USA_WIND"] = pd.to_numeric(df_ni["USA_WIND"], errors="coerce")
    df_ni["USA_PRES"] = pd.to_numeric(df_ni["USA_PRES"], errors="coerce")

    # Composite best wind and pressure
    df_ni["WIND_KTS"] = (
        df_ni["NEWDELHI_WIND"]
        .combine_first(df_ni["WMO_WIND"])
        .combine_first(df_ni["USA_WIND"])
    )
    df_ni["PRES_HPA"] = (
        df_ni["NEWDELHI_PRES"]
        .combine_first(df_ni["WMO_PRES"])
        .combine_first(df_ni["USA_PRES"])
    )

    # Filter rows with valid coordinates
    df_ni = df_ni.dropna(subset=["LAT", "LON", "ISO_TIME"]).copy()

    # Filter for seasons >= 1980 (modern satellite/observation era)
    df_modern = df_ni[df_ni["SEASON"] >= 1980].copy()

    # Map IMD intensity grade
    class_info = [
        map_imd_intensity(w, g)
        for w, g in zip(df_modern["WIND_KTS"], df_modern["NEWDELHI_GRADE"])
    ]
    df_modern["IMD_CLASS_CODE"] = [c[0] for c in class_info]
    df_modern["IMD_CLASS_NAME"] = [c[1] for c in class_info]

    # Clean storm name
    df_modern["STORM_NAME"] = df_modern["NAME"].fillna("NOT_NAMED").str.strip()

    # Subbasin naming
    subbasin_map = {"BB": "Bay of Bengal", "AS": "Arabian Sea", "MM": "Myanmar Coast"}
    df_modern["SUBBASIN_NAME"] = df_modern["SUBBASIN"].map(
        lambda x: subbasin_map.get(str(x), "North Indian Ocean")
    )

    # Select core relevant columns
    cols_to_keep = [
        "SID",
        "SEASON",
        "STORM_NAME",
        "ISO_TIME",
        "SUBBASIN",
        "SUBBASIN_NAME",
        "LAT",
        "LON",
        "WIND_KTS",
        "PRES_HPA",
        "NEWDELHI_GRADE",
        "IMD_CLASS_CODE",
        "IMD_CLASS_NAME",
        "DIST2LAND",
        "LANDFALL",
    ]
    df_clean = df_modern[cols_to_keep].sort_values(by=["SID", "ISO_TIME"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(PROCESSED_TRACKS_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_TRACKS_PATH, index=False)
    print(f"Saved processed cyclone tracks to {PROCESSED_TRACKS_PATH} ({len(df_clean):,} records).")

    # Generate Cyclone Catalog (storm-level summary)
    catalog = []
    for sid, group in df_clean.groupby("SID"):
        catalog.append(
            {
                "SID": sid,
                "SEASON": int(group["SEASON"].iloc[0]),
                "STORM_NAME": group["STORM_NAME"].iloc[0],
                "SUBBASIN": group["SUBBASIN_NAME"].iloc[0],
                "START_TIME": group["ISO_TIME"].min().isoformat(),
                "END_TIME": group["ISO_TIME"].max().isoformat(),
                "OBSERVATIONS_COUNT": len(group),
                "MAX_WIND_KTS": group["WIND_KTS"].max(),
                "MIN_PRES_HPA": group["PRES_HPA"].min(),
                "PEAK_IMD_CLASS": group["IMD_CLASS_NAME"].loc[
                    group["IMD_CLASS_CODE"].idxmax()
                ],
                "PEAK_IMD_CODE": group["IMD_CLASS_CODE"].max(),
            }
        )
    df_catalog = pd.DataFrame(catalog).sort_values(by=["SEASON", "START_TIME"], ascending=[False, False])
    df_catalog.to_csv(STORM_CATALOG_PATH, index=False)
    print(f"Saved storm catalog to {STORM_CATALOG_PATH} ({len(df_catalog)} unique cyclones).")

    # Print summary statistics
    print("\n--- DATASET SUMMARY (1980 - Present) ---")
    print(f"Total Cyclones: {len(df_catalog)}")
    print(f"Total Observations: {len(df_clean):,}")
    print("\nBreakdown by Peak IMD Intensity:")
    print(df_catalog["PEAK_IMD_CLASS"].value_counts())
    print("\nINSAT-3D Era (2014 - Present):")
    recent = df_catalog[df_catalog["SEASON"] >= 2014]
    print(f"Cyclones: {len(recent)}, Observations: {len(df_clean[df_clean['SEASON'] >= 2014]):,}")
    print("Notable Cyclones:", recent["STORM_NAME"].dropna().unique()[:15])


if __name__ == "__main__":
    process_ibtracs()


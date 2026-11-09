# Vision-Based Identification, Classification, and Prediction of Cyclones Using Satellite Data

A multimodal spatiotemporal cyclone prediction and monitoring platform focused on the **North Indian Ocean (Bay of Bengal and Arabian Sea)**.

---

## 📁 Repository Structure

```
├── .venv/                         # Python 3.11 virtual environment
├── datasets/
│   ├── raw/
│   │   ├── ibtracs/               # Authentic NOAA IBTrACS v4 NI basin dataset (26.61 MB)
│   │   ├── insat3d/               # Raw INSAT-3D Asia Sector IR1, VIS, Rapid Scan imagery
│   │   ├── insat3dr/              # INSAT-3DR data products
│   │   └── era5/                  # ERA5 physical variables
│   ├── processed/
│   │   ├── images/                # Standardized 224x224 normalized cyclone images
│   │   ├── storm_tracks.csv       # Cleaned 15,654 track records (1980–Present)
│   │   ├── storm_catalog.csv      # 474 unique North Indian Ocean cyclones catalog
│   │   ├── metadata.csv           # 14,665 observation records with multi-horizon targets
│   │   └── sequences/             # Temporal sequence windows for LSTM/ConvLSTM
│   └── splits/
│       ├── train.csv              # 12,429 records (379 cyclones, 1980–2019)
│       ├── validation.csv         # 995 records (34 cyclones, 2020–2022)
│       └── test.csv               # 1,241 records (41 cyclones, 2023–2026 unseen holdout)
├── scripts/
│   ├── download_ibtracs.py        # Automated downloader for NOAA IBTrACS NI basin
│   ├── process_ibtracs.py         # Track processor and IMD grade mapper
│   ├── fetch_satellite_imagery.py # MOSDAC RSS parser & IMD live satellite ingest
│   ├── build_metadata.py          # Unified metadata builder with forecast horizons
│   └── verify_dataset.py          # Dataset auditor and leakage verification
├── .env.example                   # Environment configuration template
└── README.md
```

---

## 🚀 Phase 1: Dataset Acquisition & Preparation (Completed)

1. **NOAA IBTrACS v4 Best-Track Dataset**:
   - Downloaded official North Indian Ocean (`NI`) best-track archive directly from NOAA NCEI.
   - Filtered into Bay of Bengal (`BB`) and Arabian Sea (`AS`) storms.
   - Cleaned coordinates, central pressure, maximum sustained wind speeds, and mapped into official **IMD intensity classes**:
     - *Depression (17–27 kt)*
     - *Deep Depression (28–33 kt)*
     - *Cyclonic Storm (34–47 kt)*
     - *Severe Cyclonic Storm (48–63 kt)*
     - *Very Severe Cyclonic Storm (64–89 kt)*
     - *Extremely Severe Cyclonic Storm (90–119 kt)*
     - *Super Cyclonic Storm (≥ 120 kt)*

2. **Live & Historical Satellite Ingestion (MOSDAC & IMD)**:
   - Automated ingestion of MOSDAC RSS feeds (`3dimager.xml` & `3drimager.xml`) extracting Level-1C & Level-2 product notifications.
   - Live download of INSAT-3D Asia Sector Infrared (`3Dasiasec_ir1.jpg`), Visible (`3Dasiasec_vis.jpg`), and Rapid Scan IR1 products.
   - Verified IMD cyclone API endpoints (`/cyclone_track`, `/cyclone_wind`, `/cyclone_cou`).

3. **Zero-Leakage Chronological & Storm-Wise Splits**:
   - Split strictly by **entire Cyclone ID (`SID`)**, preventing image frames of the same cyclone from co-occurring in train and test sets.
   - Train Set: 1980–2019 (12,429 points across 379 cyclones)
   - Validation Set: 2020–2022 (995 points across 34 cyclones)
   - Test Set: 2023–2026 Holdout (1,241 points across 41 cyclones including Mocha, Biparjoy, Tej, Hamoon, Midhili, Michaung, Remal, Dana)

---

## 🛠️ Verification & Reproducibility

To re-run and audit the dataset pipeline:

```bash
# 1. Activate environment
.\.venv\Scripts\activate

# 2. Re-download or update IBTrACS NI
python scripts/download_ibtracs.py

# 3. Process tracks and map IMD classes
python scripts/process_ibtracs.py

# 4. Fetch live satellite imagery & MOSDAC catalog
python scripts/fetch_satellite_imagery.py

# 5. Build metadata with +6h, +12h, +24h targets & splits
python scripts/build_metadata.py

# 6. Run full verification audit
python scripts/verify_dataset.py
```


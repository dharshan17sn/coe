"""
Script to download NOAA IBTrACS v4 best-track data for North Indian Ocean (NI) basin.
Basin covers Bay of Bengal and Arabian Sea.
Source: NOAA NCEI (National Centers for Environmental Information)
"""

import os
import sys
import requests
from tqdm import tqdm

IBTRACS_NI_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)
OUTPUT_DIR = os.path.join("datasets", "raw", "ibtracs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ibtracs.NI.list.v04r01.csv")


def download_ibtracs_ni():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        if file_size > 20 * 1024 * 1024:  # >20MB indicates complete download
            print(f"IBTrACS NI dataset already exists at {OUTPUT_FILE} ({file_size / (1024*1024):.2f} MB). Skipping download.")
            return OUTPUT_FILE

    print(f"Downloading IBTrACS North Indian Ocean (NI) dataset...")
    print(f"URL: {IBTRACS_NI_URL}")
    print(f"Destination: {OUTPUT_FILE}")

    response = requests.get(IBTRACS_NI_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB chunks

    with open(OUTPUT_FILE, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, desc="IBTrACS NI"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    final_size = os.path.getsize(OUTPUT_FILE)
    print(f"Successfully downloaded IBTrACS NI dataset: {final_size / (1024*1024):.2f} MB")
    return OUTPUT_FILE


if __name__ == "__main__":
    download_ibtracs_ni()

"""
Ingestion script for satellite imagery & live monitoring sources:
1. MOSDAC RSS Feeds (INSAT-3D & INSAT-3DR Imager payloads)
2. IMD Live Satellite Products (INSAT-3D Asia-Sector IR1 & Visible)
3. IMD Cyclone API Endpoint verification
"""

import os
import re
import json
import datetime
import xml.etree.ElementTree as ET
import requests

MOSDAC_3D_RSS = "https://mosdac.gov.in/3dimager.xml"
MOSDAC_3DR_RSS = "https://mosdac.gov.in/3drimager.xml"

IMD_SATELLITE_PRODUCTS = {
    "INSAT3D_ASIA_IR1": "https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg",
    "INSAT3D_ASIA_VIS": "https://mausam.imd.gov.in/Satellite/3Dasiasec_vis.jpg",
    "INSAT3D_RAPID_SCAN_IR1": "https://mausam.imd.gov.in/Satellite/rswmo_ir1.jpg",
}

IMD_CYCLONE_APIS = {
    "cyclone_track": "https://api.imd.gov.in/api/v1/cyclone_track",
    "cyclone_wind": "https://api.imd.gov.in/api/v1/cyclone_wind",
    "cyclone_cou": "https://api.imd.gov.in/api/v1/cyclone_cou",
}

OUTPUT_DIR_3D = os.path.join("datasets", "raw", "insat3d")
OUTPUT_DIR_3DR = os.path.join("datasets", "raw", "insat3dr")


def fetch_mosdac_rss(url, satellite_name):
    print(f"Fetching MOSDAC RSS feed for {satellite_name} ({url})...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15, verify=False)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            items.append({
                "satellite": satellite_name,
                "title": title,
                "link": link,
                "pub_date": pub_date,
            })
        print(f"Retrieved {len(items)} product notifications from {satellite_name} feed.")
        return items
    except Exception as e:
        print(f"Error fetching MOSDAC feed {satellite_name}: {e}")
        return []


def download_live_satellite_image(product_name, url):
    os.makedirs(OUTPUT_DIR_3D, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S")
    filename = f"{product_name}_{timestamp_str}.jpg"
    dest_path = os.path.join(OUTPUT_DIR_3D, filename)

    print(f"Downloading live satellite image [{product_name}] from {url}...")
    try:
        resp = requests.get(url, headers=headers, timeout=20, verify=False)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"Saved: {dest_path} ({size_kb:.1f} KB)")
        return {
            "product": product_name,
            "path": dest_path,
            "filename": filename,
            "timestamp": now_utc.isoformat(),
            "size_bytes": len(resp.content),
            "status": "SUCCESS",
        }
    except Exception as e:
        print(f"Failed to download {product_name}: {e}")
        return {
            "product": product_name,
            "url": url,
            "error": str(e),
            "status": "FAILED",
        }


def check_imd_cyclone_apis():
    print("\nVerifying IMD Cyclone API Endpoints...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }
    api_status = {}
    for name, url in IMD_CYCLONE_APIS.items():
        try:
            resp = requests.get(url, headers=headers, timeout=10, verify=False)
            api_status[name] = {
                "url": url,
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
                "reachable": resp.status_code in [200, 401, 403, 404],
                "note": "Endpoint active" if resp.status_code == 200 else f"HTTP {resp.status_code}"
            }
            print(f" - {name} ({url}): HTTP {resp.status_code}")
        except Exception as e:
            api_status[name] = {
                "url": url,
                "reachable": False,
                "error": str(e),
            }
            print(f" - {name}: Unreachable ({e})")
    return api_status


def run_satellite_ingestion():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(OUTPUT_DIR_3D, exist_ok=True)
    os.makedirs(OUTPUT_DIR_3DR, exist_ok=True)

    # 1. MOSDAC Feeds
    feed_3d = fetch_mosdac_rss(MOSDAC_3D_RSS, "INSAT-3D")
    feed_3dr = fetch_mosdac_rss(MOSDAC_3DR_RSS, "INSAT-3DR")
    all_feeds = feed_3d + feed_3dr

    feed_index_path = os.path.join(OUTPUT_DIR_3D, "mosdac_feeds_catalog.json")
    with open(feed_index_path, "w", encoding="utf-8") as f:
        json.dump(all_feeds, f, indent=2)
    print(f"Saved MOSDAC feeds catalog to {feed_index_path}")

    # 2. Download Live Satellite Imagery
    downloaded_images = []
    for prod_name, prod_url in IMD_SATELLITE_PRODUCTS.items():
        res = download_live_satellite_image(prod_name, prod_url)
        downloaded_images.append(res)

    images_manifest_path = os.path.join(OUTPUT_DIR_3D, "satellite_images_manifest.json")
    with open(images_manifest_path, "w", encoding="utf-8") as f:
        json.dump(downloaded_images, f, indent=2)

    # 3. IMD Cyclone API Check
    api_results = check_imd_cyclone_apis()
    api_report_path = os.path.join(OUTPUT_DIR_3D, "imd_api_status.json")
    with open(api_report_path, "w", encoding="utf-8") as f:
        json.dump(api_results, f, indent=2)

    print("\n--- INGESTION COMPLETE ---")
    print(f"Downloaded {sum(1 for img in downloaded_images if img.get('status') == 'SUCCESS')} satellite images.")
    print(f"Indexed {len(all_feeds)} MOSDAC product notifications.")


if __name__ == "__main__":
    run_satellite_ingestion()

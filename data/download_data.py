import os
import requests
import zipfile
from pathlib import Path

UPSTREAM_RELEASE_URL = "https://api.github.com/repos/ZIN99606/stock_daily_crawler/releases/latest"
DATA_DIR = Path("./data/raw")

def download_latest_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    resp = requests.get(UPSTREAM_RELEASE_URL, timeout=30)
    resp.raise_for_status()
    release = resp.json()
    
    asset = None
    for a in release.get("assets", []):
        if a["name"].startswith("data_center_full_") and a["name"].endswith(".zip"):
            asset = a
            break
    
    if not asset:
        raise ValueError("未找到数据文件")
    
    zip_path = DATA_DIR / asset["name"]
    print(f"下载: {asset['name']} ({asset['size'] // 1024 // 1024} MB)")
    
    r = requests.get(asset["url"], headers={"Accept": "application/octet-stream"}, timeout=120)
    r.raise_for_status()
    zip_path.write_bytes(r.content)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    
    print(f"解压完成: {DATA_DIR / 'data_center'}")
    return DATA_DIR / "data_center"

if __name__ == "__main__":
    download_latest_data()

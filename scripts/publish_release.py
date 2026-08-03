import os
import sys
import requests

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    print("错误: 先执行 export GITHUB_TOKEN='你的token'")
    sys.exit(1)

REPO = "meguru296/stock_transformer_forecast"
TAG = "v1.0.0"
FILE_PATH = "outputs/forecasts/forecast_20260731.parquet"

headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

url = f"https://api.github.com/repos/{REPO}/releases"
data = {"tag_name": TAG, "name": f"Forecast {TAG}", "body": "预测输出文件"}

r = requests.post(url, headers=headers, json=data)
if r.status_code == 422:
    r = requests.get(f"{url}/tags/{TAG}", headers=headers)
elif r.status_code != 201:
    print(f"创建失败: {r.status_code}")
    print(r.text)
    sys.exit(1)

release = r.json()
upload_url = release["upload_url"].replace("{?name,label}", "")
filename = os.path.basename(FILE_PATH)

with open(FILE_PATH, "rb") as f:
    r = requests.post(
        upload_url,
        headers={**headers, "Content-Type": "application/octet-stream"},
        params={"name": filename},
        data=f
    )

if r.status_code == 201:
    print(f"发布成功: {r.json()['browser_download_url']}")
else:
    print(f"上传失败: {r.status_code}")
    print(r.text)

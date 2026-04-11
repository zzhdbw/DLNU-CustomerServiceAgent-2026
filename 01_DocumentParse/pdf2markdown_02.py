# https://mineru.net/

import os
import requests
from dotenv import load_dotenv
import requests
import zipfile
import os
import shutil

load_dotenv()
token = os.environ["API_TOKEN"]

processed_zip_dir = "data/processed_zip"
processed_dir = "data/processed"

task_id = "50b2049f-b36f-4b54-9583-175f952f31b0"
url = f"https://mineru.net/api/v4/extract/task/{task_id}"
header = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

res = requests.get(url, headers=header)

full_zip_url = res.json()["data"]["full_zip_url"]

os.makedirs(processed_zip_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

zip_filename = os.path.join(processed_zip_dir, f"{task_id}.zip")
response = requests.get(full_zip_url, stream=True)
with open(zip_filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

extract_dir = os.path.join(processed_dir, task_id)
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

with zipfile.ZipFile(zip_filename, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"下载完成: {zip_filename}")
print(f"解压完成: {extract_dir}")

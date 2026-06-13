import os
import re
from glob import glob

import requests
from dotenv import load_dotenv
from pymilvus import MilvusClient
from tqdm import tqdm

load_dotenv()

DATA_DIR = "data/phone_docs/zh"


def read_md() -> list[str]:
    """
    读取所有手机相关 Markdown 文件，按 H1/H2 切分，子节自动带上父标题上下文。
    """
    text_lines = []

    for file_path in glob(f"{DATA_DIR}/**/*.md", recursive=True):
        with open(file_path, "r") as file:
            content = file.read()

        raw_chunks = re.split(r"\n(?=##?\s)", f"\n{content}")
        parent_title = ""
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            title = chunk.split("\n")[0].strip()

            if title.startswith("# ") or title.startswith("#\u3000"):
                parent_title = title
                body = "\n".join(chunk.split("\n")[1:]).strip()
                if body:
                    text_lines.append(chunk)
            else:
                enriched = f"{parent_title}\n\n{chunk}"
                text_lines.append(enriched)

    return text_lines


def emb_text(text: str) -> list[float]:
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '你的Jina API Key')}",
    }
    data = {
        "model": "jina-embeddings-v5-text-small",
        "task": "retrieval.query",
        "normalized": True,
        "input": [
            text,
        ],
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()["data"][0]["embedding"]


def create_db(db_path: str, collection_name: str, dimension: int = 1024) -> MilvusClient:
    milvus_client = MilvusClient(uri=db_path)

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        metric_type="IP",  # Inner product distance
        consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/tune_consistency.md#Consistency-Level for more details.
    )

    return milvus_client


def create_emb(text_lines: list[str]) -> list[dict]:
    """
    创建嵌入向量
    """
    data = []

    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(line), "text": line})

    return data


if __name__ == "__main__":
    collection_name = "my_rag_collection"
    # 读取数据
    text_lines = read_md()
    # 创建数据库
    milvus_client = create_db(db_path="./db_files/milvus_demo.db", collection_name=collection_name)
    # 创建嵌入向量
    data = create_emb(text_lines)
    # 插入数据库
    d = milvus_client.insert(collection_name=collection_name, data=data)
    print(d)

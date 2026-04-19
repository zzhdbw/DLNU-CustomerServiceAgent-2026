import os
from tqdm import tqdm
from pymilvus import MilvusClient
from glob import glob
import json
import requests

# 先获取数据
# wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
# unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs


def read_md() -> list[str]:
    """
    读取所有Markdown文件，返回一个包含所有文本的列表。
    """
    text_lines = []

    for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")
    return text_lines


def emb_text(text: str) -> list[float]:

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer jina_dfed9a88f2de4aee9c3b20e9ca69bc5f6rHHv1iWegzpYxPN6haaL49mch5l",
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

    return response.json()["data"][0]['embedding']


def create_db(
    db_path: str, collection_name: str, dimension: int = 1024
) -> MilvusClient:
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
    milvus_client = create_db(
        db_path="./db_files/milvus_demo.db", collection_name=collection_name
    )
    # 创建嵌入向量
    data = create_emb(text_lines)
    # 插入数据库
    d = milvus_client.insert(collection_name=collection_name, data=data)
    print(d)

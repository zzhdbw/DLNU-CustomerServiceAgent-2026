from pymilvus import MilvusClient
import random
from pprint import pprint
import numpy as np


# 01 创建/连接向量数据库
client = MilvusClient("db_files/milvus_demo.db")

# 02 创建表/集合
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")

client.create_collection(
    collection_name="demo_collection",
    dimension=64,  # The vectors we will use in this demo has 64 dimensions
)

# 03 数据准备
history_docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
# Use fake representation with random vectors (64 dimension).
vectors = np.array([np.random.uniform(-1, 1, 64) for _ in history_docs])
data = [
    {"id": i, "vector": vectors[i], "text": history_docs[i], "subject": "history"}
    for i in range(len(vectors))
]

pprint(data)

# 04 插入数据
res = client.insert(collection_name="demo_collection", data=data)
print(res)

# 05 搜索向量
query_vectors = [[random.uniform(-1, 1) for _ in range(64)]]

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    filter="subject == 'history'",  # 元数据过滤
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)

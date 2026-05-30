from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction


# 1. 连接 Milvus Lite
client = MilvusClient(uri="./milvus_hybrid_demo.db")

collection_name = "hybrid_dense_bm25_demo"

if client.has_collection(collection_name):
    client.drop_collection(collection_name)


# 2. 准备 dense embedding 模型
# 你也可以换成 BGE、OpenAI embedding 等
dense_ef = SentenceTransformerEmbeddingFunction(
    model_name="../03_DenseRetrieval/models/BAAI/bge-m3", device="cpu"
)

dense_dim = dense_ef.dim


# 3. 定义 schema
schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

# 原始文本字段：BM25 Function 会从这里读取文本
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=4096,
    enable_analyzer=True,  # 关键：启用 analyzer，供全文检索/BM25 使用
)

# dense 向量字段
schema.add_field(
    field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim
)

# sparse 向量字段：BM25 Function 的输出字段
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)


# 4. 添加 BM25 Function
# input_field_names: 文本字段
# output_field_names: 自动生成的 BM25 sparse vector 字段
bm25_function = Function(
    name="text_bm25_emb",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse_vector"],
)

schema.add_function(bm25_function)


# 5. 创建索引
index_params = client.prepare_index_params()

# dense 索引
index_params.add_index(
    field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE"
)

# sparse BM25 索引
index_params.add_index(
    field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
)


# 6. 创建 collection
client.create_collection(
    collection_name=collection_name, schema=schema, index_params=index_params
)


# 7. 插入数据
docs = [
    "Milvus 是一个面向 AI 应用的向量数据库，支持向量相似度检索。",
    "BM25 是一种经典的关键词相关性排序算法，常用于全文检索。",
    "混合检索结合了稠密向量语义检索和 BM25 稀疏关键词检索。",
    "BGE-M3 支持生成 dense embedding、sparse lexical weights 和 multi-vector 表示。",
    "在 RAG 系统中，混合检索可以提升专业术语、编号、实体名的召回效果。",
]

dense_vectors = dense_ef.encode_documents(docs)

data = [
    {
        "text": text,
        "dense_vector": dense_vector,
        # 注意：不用传 sparse_vector
        # Milvus 会通过 BM25 Function 自动生成
    }
    for text, dense_vector in zip(docs, dense_vectors)
]

client.insert(collection_name=collection_name, data=data)

client.load_collection(collection_name)


# 8. 构造查询
query = "Milvus 怎么结合 BM25 做混合检索？"

query_dense_vector = dense_ef.encode_queries([query])[0]


# 9. dense 检索请求
dense_request = AnnSearchRequest(
    data=[query_dense_vector],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {}},
    limit=5,
)


# 10. sparse BM25 检索请求
# 注意：BM25 Function 场景下，query 直接传原始文本即可
sparse_request = AnnSearchRequest(
    data=[query],
    anns_field="sparse_vector",
    param={"metric_type": "BM25", "params": {}},
    limit=5,
)


# 11. 混合检索：dense + BM25 sparse，然后用 RRF 融合
results = client.hybrid_search(
    collection_name=collection_name,
    reqs=[dense_request, sparse_request],
    ranker=RRFRanker(),
    limit=3,
    output_fields=["text"],
)


# 12. 打印结果
for hits in results:
    for hit in hits:
        print("score:", hit["distance"])
        print("text:", hit["entity"]["text"])
        print("---")

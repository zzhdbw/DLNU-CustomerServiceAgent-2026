import os
from pprint import pprint

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter

DATA_DIR = "data/phone_docs/zh/"
DB_PATH = "db_files/phone_qa.db"
COLLECTION_NAME = "phone_qa_collection"
DIMENSION = 768

# 步骤1：加载所有 MD 文件
loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.md",
    show_progress=True,
    loader_cls=TextLoader,  # MD文件必须指定，否则会失去标题层级上下文
)
raw_docs = loader.load()

# 步骤2：按 H1/H2/H3 切分，保留标题层级上下文
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
)

result_chunks = []
for doc in raw_docs:  # 遍历每个文档
    chunks = splitter.split_text(doc.page_content)  # 对文档内容进行切分

    for chunk in chunks:  # 遍历每一个切片
        # 确保所有标题字段都有值，否则 Milvus schema 会因字段不一致报错
        chunk.metadata["H1"] = chunk.metadata.get("H1", "")
        chunk.metadata["H2"] = chunk.metadata.get("H2", "")
        chunk.metadata["H3"] = chunk.metadata.get("H3", "")
        # 提取标题层级信息，拼接到原文中
        header_info = f"{chunk.metadata['H1']}\n{chunk.metadata['H2']}\n{chunk.metadata['H3']}\n".lstrip("\n")
        chunk.page_content = header_info + chunk.page_content
        # 保留文档来源
        chunk.metadata["source"] = doc.metadata.get("source", "")
        # 添加至结果列表
        result_chunks.append(chunk)

# print(result_chunks)

# 步骤3：创建向量模型
embeddings = JinaEmbeddings(
    model="jina-embeddings-v5-text-small",
    jina_api_key=os.getenv("JINA_API_KEY", "你的Jina API Key"),
)
# 步骤4：创建向量库对象，数据库不存在就创建，已有就在原有基础上添加
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": DB_PATH},
    collection_name=COLLECTION_NAME,
    auto_id=True,
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
)
# index_type（索引类型）：
# - FLAT — 暴力全量搜索，精度最高，小数据集首选
# - IVF_FLAT — 倒排索引，需要设 nlist（如 {"nlist": 128}），大数据集加速
# - IVF_SQ8 — IVF + 量化压缩，更省内存但精度略降
# - HNSW — 图索引，性能好，需设 M 和 efConstruction
# - DISKANN — 磁盘索引，适合内存放不下的超大数据集
# metric_type（距离度量）：
# - L2 — 欧氏距离，值越小越相似（默认）
# - IP — 内积，值越大越相似
# - COSINE — 余弦相似度，值越大越相似

# 步骤5：添加文档到向量库，
result = vector_store.add_documents(documents=result_chunks)
pprint(result)

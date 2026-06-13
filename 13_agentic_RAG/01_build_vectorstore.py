import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

DATA_DIR = "./data/phone_docs/zh/"
DB_PATH = "db_files/phone_qa.db"
COLLECTION_NAME = "phone_qa_collection"

# 步骤1：加载所有 MD 文件
loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.md",
    show_progress=True,
    loader_cls=TextLoader,
)
raw_docs = loader.load()

# 步骤2：按 H1/H2/H3 切分，保留标题层级上下文
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
)

result_chunks = []
for doc in raw_docs:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        chunk.metadata["H1"] = chunk.metadata.get("H1", "")
        chunk.metadata["H2"] = chunk.metadata.get("H2", "")
        chunk.metadata["H3"] = chunk.metadata.get("H3", "")
        header_info = f"{chunk.metadata['H1']}\n{chunk.metadata['H2']}\n{chunk.metadata['H3']}\n".lstrip("\n")
        chunk.page_content = header_info + chunk.page_content
        chunk.metadata["source"] = doc.metadata.get("source", "")
        result_chunks.append(chunk)

# 步骤3：创建向量模型
embeddings = JinaEmbeddings(
    model="jina-embeddings-v5-text-small",
    jina_api_key=os.getenv("JINA_API_KEY", "你的Jina API Key"),
)

# 步骤4：创建向量库对象
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": DB_PATH},
    collection_name=COLLECTION_NAME,
    auto_id=True,
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
)

# 步骤5：添加文档到向量库
result = vector_store.add_documents(documents=result_chunks)
pprint(result)

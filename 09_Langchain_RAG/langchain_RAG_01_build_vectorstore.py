import re
from glob import glob
from tqdm import tqdm
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_community.embeddings import JinaEmbeddings

DATA_DIR = "data/phone_docs/zh"
DB_PATH = "db_files/phone_qa.db"
COLLECTION_NAME = "phone_qa_collection"
DIMENSION = 768  # jina-embeddings-v5-text-small 实际输出


def load_and_split_md(data_dir: str) -> list[Document]:
    """读取所有 MD 文件，按 H1/H2 切分，子节自动带上父标题上下文"""
    docs = []
    for file_path in glob(f"{data_dir}/**/*.md", recursive=True):
        with open(file_path, "r") as f:
            content = f.read()

        raw_chunks = re.split(r'\n(?=##?\s)', f'\n{content}')
        parent_title = ""
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            lines = chunk.split("\n")
            title = lines[0].strip()

            # 追踪当前 H1 标题
            if title.startswith("# ") or title.startswith("#　"):
                parent_title = title
                # 纯标题无正文的 H1 块（如只有 "# 常见使用问题"）跳过，避免污染检索
                body = "\n".join(lines[1:]).strip()
                if body:
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": file_path, "title": title},
                        )
                    )
            else:
                # H2 子节 — 在前面加上父 H1 标题作为上下文
                enriched = f"{parent_title}\n\n{chunk}"
                docs.append(
                    Document(
                        page_content=enriched,
                        metadata={"source": file_path, "title": title},
                    )
                )
    return docs


def build_vectorstore(docs: list[Document]):
    """构建 Milvus 向量库（使用 LangChain Embeddings + MilvusClient）"""
    embeddings = JinaEmbeddings(
        model="jina-embeddings-v5-text-small",
        jina_api_key="jina_dfed9a88f2de4aee9c3b20e9ca69bc5f6rHHv1iWegzpYxPN6haaL49mch5l",
    )

    # 创建/清空数据库
    milvus_client = MilvusClient(uri=DB_PATH)
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=DIMENSION,
        metric_type="IP",
        consistency_level="Bounded",
    )

    # 批量生成向量（LangChain embed_documents 自动做 batch 调用）
    texts = [doc.page_content for doc in docs]
    print(f"   正在批量生成 {len(texts)} 条向量...")
    vectors = embeddings.embed_documents(texts)

    # 批量插入 Milvus
    data = []
    for i, doc in enumerate(tqdm(docs, desc="Inserting into Milvus")):
        data.append(
            {
                "id": i,
                "vector": vectors[i],
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
            }
        )

    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"   插入完成")

    print(f"✅ 向量库创建完成，共 {len(docs)} 个文档片段")
    return milvus_client


if __name__ == "__main__":
    print("📖 读取并切分文档...")
    docs = load_and_split_md(DATA_DIR)
    print(f"   共 {len(docs)} 个片段")

    print("🔧 构建向量数据库（LangChain JinaEmbeddings + MilvusClient）...")
    build_vectorstore(docs)
    print(f"   数据库路径: {DB_PATH}")

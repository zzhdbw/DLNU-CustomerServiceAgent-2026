from langchain_milvus import Milvus
from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

DB_PATH = "db_files/phone_qa.db"
COLLECTION_NAME = "phone_qa_collection"
TOP_K = 10

SYSTEM_PROMPT = "你是一个手机商城智能客服助手，请根据提供的产品资料回答用户的售前咨询和售后问题。回答需简洁专业、热情友好，优先从资料中提取信息。如果资料没有相关信息，可以基于常见行业常识补充，但需说明'资料未提及，以下为一般情况供参考'。"
USER_PROMPT = """请根据 <context> 标签中的资料回答 <question> 标签中的问题。
<context>
{context}
</context>
<question>
{question}
</question>

请用中文回答问题"""

# 初始化嵌入模型
embeddings = JinaEmbeddings(
    model="jina-embeddings-v5-text-small",
    jina_api_key=os.getenv("JINA_API_KEY", "你的Jina API Key"),
)

# 初始化向量数据库
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": DB_PATH},
    collection_name=COLLECTION_NAME,
)

# 创建 LLM 实例
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com/v1",
)


def search_milvus(question: str) -> list[dict]:
    """在 Milvus 中检索与 question 最相似的文档片段"""

    docs = vector_store.similarity_search(question, k=TOP_K)

    results = []
    for doc in docs:
        results.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
            }
        )
    return results


def build_context(results: list[dict]) -> str:
    return "\n\n".join(r["text"] for r in results)


def build_rag_chain():
    """构建 LangChain RAG 链路（不含检索器，调用前需传入 context）"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }  # 代表将context和question透传给prompt，之后调用chain时需要传入context和question
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    question = "小米15 16GB + 512GB多少钱？"
    print(f"问题: {question}\n")

    # 构建大模型调用链
    chain = build_rag_chain()

    use_RAG = False
    if use_RAG:
        print("🔍 检索中...")
        results = search_milvus(question)
        print(f"命中 {len(results)} 条")
        context = build_context(results)
    else:
        context = "没有相关资料"

    print("\n💡 回答:")
    answer = chain.invoke({"context": context, "question": question})
    print(answer)

from pymilvus import MilvusClient
from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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


def search_milvus(question: str) -> list[dict]:
    """在 Milvus 中检索与 question 最相似的文档片段"""
    embeddings = JinaEmbeddings(
        model="jina-embeddings-v5-text-small",
        jina_api_key="jina_dfed9a88f2de4aee9c3b20e9ca69bc5f6rHHv1iWegzpYxPN6haaL49mch5l",
    )

    milvus_client = MilvusClient(uri=DB_PATH)
    if not milvus_client.has_collection(COLLECTION_NAME):
        return []

    question_vector = embeddings.embed_query(question)

    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[question_vector],
        limit=TOP_K,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    results = []
    for hit in search_res[0]:
        results.append(
            {
                "text": hit["entity"]["text"],
                "distance": round(hit["distance"], 4),
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

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-c576413004a44dfeb327d8431b612bcb",
        base_url="https://api.deepseek.com/v1",
    )

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    question = "小米14多少钱？支持分期吗？"
    print(f"❓ 问题: {question}\n")

    print("🔍 检索中...")
    results = search_milvus(question)
    print(f"   命中 {len(results)} 条")

    context = build_context(results)
    chain = build_rag_chain()

    print("\n💡 回答:")
    answer = chain.invoke({"context": context, "question": question})
    print(answer)

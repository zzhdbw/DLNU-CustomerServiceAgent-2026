import os
from operator import itemgetter

import gradio as gr
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI

DB_PATH = "db_files/phone_qa.db"
COLLECTION_NAME = "phone_qa_collection"

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
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
)

# 创建 LLM 实例
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com/v1",
)


def search_milvus(question: str, top_k: int = 10) -> list[dict]:
    """在 Milvus 中检索与 question 最相似的文档片段，返回内容及相似度"""

    docs_with_score = vector_store.similarity_search_with_score(question, k=top_k)

    results = []
    for doc, score in docs_with_score:
        results.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "score": round(score, 4),
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
            MessagesPlaceholder(variable_name="history"),  # 历史占位符
            ("human", USER_PROMPT),
        ]
    )

    chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def chat(question, history, use_rag, top_k):
    # 构建大模型调用链
    chain = build_rag_chain()

    # 将历史对话转为 LangChain 消息列表
    history_messages = []
    for message in history:
        if message["role"] == "user":
            history_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history_messages.append(AIMessage(content=message["content"]))

    if use_rag:
        results = search_milvus(question, top_k)
        context = build_context(results)

        # 构建检索结果展示部分
        retrieval_info = "\n\n<details><summary>📎 召回的文档片段</summary>\n\n"
        for r in results:
            source = r["source"].replace("data/phone_docs/zh/", "").replace(".md", "")
            similarity = round(1 - r["score"], 4)  # r["score"]是两个向量的距离，相似度是1-距离
            retrieval_info += (
                f"**来源:** {source} &nbsp;|&nbsp; **相似度:** `{similarity}`\n\n> {r['text'][:200]}...\n\n---\n\n"
            )
        retrieval_info += "</details>"
    else:
        context = "没有相关资料"
        retrieval_info = ""

    # 流式输出回答
    answer = ""
    for chunk in chain.stream({"context": context, "question": question, "history": history_messages}):
        answer += chunk
        yield answer

    # 回答之后展示召回内容
    if retrieval_info:
        yield answer + retrieval_info


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        additional_inputs=[
            gr.Checkbox(label="启用 RAG 检索", value=True),
            gr.Slider(minimum=1, maximum=10, step=1, value=10, label="检索数量 TopK"),
        ],
        title="手机客服助手",
        description="基于 RAG 的智能客服系统，可查询手机规格、价格等信息。",
        examples=[
            ["小米15 16GB + 512GB多少钱？", True, 10],
            ["华为P60支持无线充电吗？", True, 10],
            ["红米K70的屏幕参数是什么？", True, 10],
            ["华为P60和小米14哪个更值得买？", True, 10],
        ],
    ).launch(server_name="0.0.0.0", server_port=7863)

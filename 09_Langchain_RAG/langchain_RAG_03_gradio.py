"""
LangChain RAG — Gradio 流式对话界面
- LangChain: JinaEmbeddings + ChatOpenAI(DeepSeek) + ChatPromptTemplate
- LangChain LCEL 流式输出
- Gradio ChatInterface 对话 UI
"""

import gradio as gr
from pymilvus import MilvusClient
from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 配置 ──────────────────────────────────────────────────────────────
DB_PATH = "09_Langchain_RAG/db_files/phone_qa.db"
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
        results.append({
            "text": hit["entity"]["text"],
            "distance": round(hit["distance"], 4),
        })
    return results


def build_context_display(results: list[dict]) -> str:
    """构建可折叠的检索结果展示 HTML"""
    lines = ["<details><summary>📄 检索结果（点击展开）</summary>\n\n"]
    for i, r in enumerate(results, 1):
        snippet = r["text"][:200].replace("\n", " ") + (
            "..." if len(r["text"]) > 200 else ""
        )
        lines.append(f"**#{i}** (相似度: {r['distance']})\n\n> {snippet}\n\n")
    lines.append("</details>")
    return "".join(lines)


def build_context(results: list[dict]) -> str:
    return "\n\n".join(r["text"] for r in results)


# 构建 LangChain LLM 链路（可复用，支持流式）
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-c576413004a44dfeb327d8431b612bcb",
    base_url="https://api.deepseek.com/v1",
)

# 流式 RAG 链路（纯 LLM 部分，检索结果在外层注入）
rag_stream = (
    {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
)


def respond(message: str, history: list):
    """RAG 流式对话函数 — 供 gr.ChatInterface 使用"""
    if not message or not message.strip():
        yield "⚠️ 请输入问题"
        return

    # ── 1. 检索 ──────────────────────────────────────────────────────
    results = search_milvus(message)
    if not results:
        yield "❌ 未检索到相关文档，请检查向量数据库"
        return

    context = build_context(results)
    context_display = build_context_display(results)

    # ── 2. 首帧：展示检索结果 ─────────────────────────────────────────
    full_response = context_display + "\n\n"
    yield full_response

    # ── 3. 流式调用 LangChain LCEL ───────────────────────────────────
    for chunk in rag_stream.stream({"context": context, "question": message}):
        full_response += chunk
        yield full_response


def build_demo():
    with gr.Blocks(title="LangChain RAG 对话", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📱 手机商城智能客服

            小米 / 华为手机 **售前咨询** + **售后支持** 智能问答系统。
            基于 **LangChain** + **Milvus**（向量检索）+ **Jina Embedding** + **DeepSeek Chat** 构建。
            """
        )

        gr.ChatInterface(
            fn=respond,
            type="messages",
            title="",
            description="",
            textbox=gr.Textbox(
                placeholder="输入问题，例如：小米14多少钱？Mate 60 Pro支持卫星通话吗？",
                container=False,
                scale=7,
            ),
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)

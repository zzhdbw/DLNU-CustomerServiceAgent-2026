import gradio as gr
from openai import OpenAI
from pymilvus import MilvusClient
from naive_RAG_01_make_embedding import emb_text

# ── 配置 ──────────────────────────────────────────────────────────────
DB_PATH = "./db_files/milvus_demo.db"
COLLECTION_NAME = "my_rag_collection"
TOP_K = 3

SYSTEM_PROMPT = """
    你是一个智能客服助手，请根据提供的上下文片段回答问题。
    """

USER_PROMPT = """
请根据 <context> 标签中的资料回答 <question> 标签中的问题。
<context>
{}
</context>
<question>
{}
</question>

请用中文回答问题
"""


def search_milvus(question: str):
    """在 Milvus 中检索与 question 最相似的文档片段"""
    milvus_client = MilvusClient(uri=DB_PATH)

    if not milvus_client.has_collection(COLLECTION_NAME):
        return [], "❌ 数据库不存在，请先运行 naive_RAG_01_make_embedding.py 创建向量库"

    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[emb_text(question)],
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
    return results, None


def build_context(results: list) -> str:
    """拼接纯文本上下文供 LLM 使用"""
    return "\n".join(r["text"] for r in results)


def build_context_display(results: list) -> str:
    """构建可折叠的检索结果展示 HTML"""
    lines = ["<details><summary>📄 检索结果（点击展开）</summary>\n\n"]
    for i, r in enumerate(results, 1):
        snippet = r["text"][:200].replace("\n", " ") + (
            "..." if len(r["text"]) > 200 else ""
        )
        lines.append(f"**#{i}** (相似度: {r['distance']})\n\n> {snippet}\n\n")
    lines.append("</details>")
    return "".join(lines)


def respond(message: str, history: list):
    """RAG 流式对话函数 — 供 gr.ChatInterface 使用"""
    if not message or not message.strip():
        yield "⚠️ 请输入问题"
        return

    # ── 1. 检索 ──────────────────────────────────────────────────────
    results, err = search_milvus(message)
    if err:
        yield err
        return

    context = build_context(results)
    context_display = build_context_display(results)

    # ── 2. 首帧：展示检索结果 ─────────────────────────────────────────
    full_response = context_display + "\n\n"
    yield full_response

    # ── 3. 流式调用 LLM ──────────────────────────────────────────────
    client = OpenAI(
        api_key="*",
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context, message)},
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield full_response


def build_demo():
    with gr.Blocks(title="Naive RAG 对话", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📚 Naive RAG 问答系统

            基于 **Milvus**（向量检索）+ **Jina Embedding** + **DeepSeek Chat** 的简易 RAG 系统。
            输入问题后自动检索知识库，流式生成回答。
            """
        )

        gr.ChatInterface(
            fn=respond,
            type="messages",
            title="",
            description="",
            textbox=gr.Textbox(
                placeholder="输入问题，例如：How is data stored in milvus?",
                container=False,
                scale=7,
            ),
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

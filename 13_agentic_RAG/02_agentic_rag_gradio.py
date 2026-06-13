"""
13_agentic_RAG: 基于 LangChain ReAct Agent + RAG 的手机客服与天气查询智能体

功能：
  1. 通过 RAG 检索手机产品资料，回答售前咨询和售后问题
  2. 查询指定城市的实时天气
  3. 查询当前系统时间

使用方式：
  cd 13_agentic_RAG
  python 02_agentic_rag_gradio.py
"""

import os
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from tianqi import get_weather as _get_weather

load_dotenv()

# ======================== 配置 ========================
DB_PATH = "db_files/phone_qa.db"
COLLECTION_NAME = "phone_qa_collection"
TOP_K = 5

# ======================== 初始化组件 ========================
embeddings = JinaEmbeddings(
    model="jina-embeddings-v5-text-small",
    jina_api_key=os.getenv("JINA_API_KEY", "你的Jina API Key"),
)

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": DB_PATH},
    collection_name=COLLECTION_NAME,
)

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
)


# ======================== 工具函数 ========================
def search_milvus(question: str, top_k: int = TOP_K) -> list[dict]:
    """在 Milvus 中检索与 question 最相似的文档片段"""
    docs = vector_store.similarity_search(question, k=top_k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
        }
        for doc in docs
    ]


@tool
def search_phone_docs(question: str) -> str:
    """搜索手机产品资料库，查询手机规格、价格、参数、售后等信息。输入为用户的自然语言问题"""
    results = search_milvus(question)
    if not results:
        return "未在资料库中找到相关信息。"
    return "\n\n".join(r["text"] for r in results)


@tool
def get_weather(city_name: str) -> str:
    """获取指定城市的天气信息，用户需要提供城市名称，如大连或大连市"""
    return _get_weather(city_name)


@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


# ======================== 创建 ReAct Agent ========================
tools = [search_phone_docs, get_weather, get_current_time]

SYSTEM_PROMPT = (
    "你是一个手机商城智能客服助手，可以回答用户的售前咨询和售后问题，也可以查询天气信息。"
    "你有以下工具可用：\n"
    "1. search_phone_docs: 搜索手机产品资料库，查询手机规格、价格、参数、售后政策等\n"
    "2. get_weather: 查询指定城市的实时天气\n"
    "3. get_current_time: 获取当前系统时间\n\n"
    "请根据用户的问题合理选择工具。回答需简洁专业、热情友好，优先从资料中提取信息。"
    "如果资料没有相关信息，可以基于常见行业常识补充，但需说明'资料未提及，以下为一般情况供参考'。"
)

agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


# ======================== Gradio 聊天界面 ========================
def chat(question: str, history: list[dict]):
    """处理用户消息，流式返回智能体思考过程与回复（最终回答逐 token 流式输出）"""
    messages = []
    for h in history:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=question))

    steps = []
    answer_so_far = ""

    # Phase 1: 流式获取工具调用过程 + 模型最终回答的逐 token 输出
    for chunk, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")

        if node == "model":
            # 模型决定调用工具（tool_call_chunks 中的第一片段带工具名）
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    name = tc.get("name") or ""
                    if name:
                        steps.append(f"<details><summary>🔧 调用工具: <code>{name}()</code></summary></details>")
                        yield "\n\n".join(steps)

            # 最终回答逐 token 输出
            if chunk.content:
                answer_so_far += chunk.content
                prefix = "\n\n".join(steps)
                yield ((prefix + "\n\n💬 " + answer_so_far) if steps else answer_so_far)

        elif node == "tools":
            # 工具返回结果
            content = str(chunk.content)
            if len(content) > 300:
                content = content[:300] + "..."
            steps.append(f"<details><summary>📎 工具返回</summary>\n\n```\n{content}\n```\n\n</details>")
            yield "\n\n".join(steps)


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        title="智能客服助手",
        description="基于 ReAct Agent + RAG 的智能客服系统，可查询手机规格、价格、天气等信息。",
        examples=[
            "小米15 16GB + 512GB多少钱？",
            "华为P60支持无线充电吗？",
            "杭州的天气怎么样？",
            "现在几点了？",
            "红米K70的屏幕参数是什么？",
        ],
    ).launch(server_name="0.0.0.0", server_port=7865)

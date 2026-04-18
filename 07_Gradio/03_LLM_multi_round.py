import os
import gradio as gr
from openai import OpenAI

SYSTEM_PROMPT = "你是一个有帮助的助手。"  # 可自定义系统提示词

# 初始化客户端
client = OpenAI(
    api_key="sk-23f015edb9a94a7f83f3c1a85753e976",  # 替换为你的DeepSeek API Key
    base_url="https://api.deepseek.com",  # DeepSeek API端点
)


def chat_with_llm(message: str, history: list):
    """
    处理用户消息，调用大模型 API 并返回回复。

    参数:
        message: 用户当前输入的消息（字符串）
        history: Gradio 对话历史，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    返回:
        模型的回复文本（字符串）
    """
    # 构建消息列表：先加入系统提示词，再加入历史对话
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 将 Gradio 历史格式转换为 API 所需格式
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    # 添加当前用户消息
    messages.append({"role": "user", "content": message})

    try:
        # 调用大模型 API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            stream=False,  # 若需流式输出，可改为 True，并使用 yield 逐步返回
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        # 错误处理：返回友好的错误提示
        return f"❌ 请求失败：{str(e)}"


# ==================== 创建 Gradio 界面 ====================
# 使用 ChatInterface 快速构建多轮对话界面
demo = gr.ChatInterface(
    fn=chat_with_llm,
    title="🤖 大模型多轮对话",
    description=f"deepseek-chat",
    examples=["介绍一下你自己", "用 Python 写一个快速排序", "今天天气怎么样？"],
    cache_examples=False,  # 对话示例不缓存，确保每次都是实时调用
)

# ==================== 启动服务 ====================
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,
        share=False,  # 如需公网临时链接，改为 True
    )

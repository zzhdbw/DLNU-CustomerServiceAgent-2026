import json
from datetime import datetime
from openai import OpenAI
from pprint import pprint

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com",
)


# ============================================================
# 真正执行的函数
# ============================================================
from tianqi import get_weather
import os


def get_current_time() -> str:
    """返回当前系统时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def search_faq(keyword: str) -> str:
    """在常见问题库中搜索"""
    faq_db = {
        "退货": "7天无理由退货，商品需保持原包装完好，运费由买家承担。",
        "发货": "下单后48小时内发货，节假日顺延，物流单号可在订单详情页查看。",
        "保修": "手机主机享有一年保修，配件（充电器、数据线）保修6个月。",
        "发票": "下单时可选择开具电子发票或纸质发票，发票随货发出。",
    }
    for k, v in faq_db.items():
        if k in keyword:
            return v
    return f"未找到关于「{keyword}」的FAQ，建议联系人工客服。"


# 函数调度表
available_functions = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "search_faq": search_faq,
}

# ============================================================
# Tool 定义
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息，用户需要提供城市名称, 如大连或大连市",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "城市名称，例如 大连市",
                    },
                },
                "required": ["city_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "搜索常见问题库，用户应提供关键词，如退货、发货、保修等",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["keyword"],
            },
        },
    },
]


# ============================================================
# Agent 循环：自动调度 tool call 并执行真正函数
# ============================================================
def agent_loop(messages: list) -> str:
    """循环直到模型生成最终文本回复（不再调用工具）"""
    while True:
        response = client.chat.completions.create(
            model="deepseek-v4-flash", messages=messages, tools=tools
        )
        message = response.choices[0].message

        if message.tool_calls:
            # 模型要调工具：把 assistant 消息加入上下文
            messages.append(message)
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🔧 调用函数: {func_name}({func_args})")

                # 真正执行函数
                func = available_functions[func_name]
                result = func(**func_args)

                print(f"   返回结果: {result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            # 最终文本回复
            return message.content


if __name__ == "__main__":
    # 测试1：单工具调用
    print("=" * 50)
    print("测试1：查询天气")
    messages = [{"role": "user", "content": "杭州的天气怎么样？"}]
    answer = agent_loop(messages)
    print(f"模型回复: {answer}")

    # 测试2：可能触发多工具
    print("\n" + "=" * 50)
    print("测试2：退货咨询")
    messages = [{"role": "user", "content": "我买了手机不满意，可以退货吗？"}]
    answer = agent_loop(messages)
    print(f"模型回复: {answer}")

    # 测试3：不触发工具的问题
    print("\n" + "=" * 50)
    print("测试3：闲聊（不触发工具）")
    messages = [{"role": "user", "content": "你好，请问你是谁？"}]
    answer = agent_loop(messages)
    print(f"模型回复: {answer}")

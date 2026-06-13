from datetime import datetime
from openai import OpenAI
import json

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-v4-flash"


from tianqi import get_weather
import os


def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def search_faq(keyword: str) -> str:
    """搜索常见问题库，用户应提供关键词，如退货、发货、保修等"""
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


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "search_faq": search_faq,
}


TOOLS = [
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
                    }
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
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "搜索常见问题库",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，如退货、发货、保修",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是一个智能客服助手，你可以使用工具来获取实时信息以回答用户问题。"
    "请根据用户的问题合理选择工具，如果不需要工具则直接回答。"
)


def run_agent(user_input: str, max_turns: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # 没有 tool call → 最终答案
        if not msg.tool_calls:
            return msg.content

        # Act: 执行每个工具调用
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments

            # 解析参数并调用对应的 Python 函数
            args = json.loads(fn_args) if fn_args else {}
            fn = TOOL_FUNCTIONS[fn_name]
            result = fn(**args)

            # Observe: 将结果追加回消息列表
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "已达到最大迭代次数，请重试。"


if __name__ == "__main__":
    # 测试1：单工具调用
    print("=" * 50)
    print("测试1：查询天气")
    print(f"模型回复: {run_agent('杭州的天气怎么样？')}")

    # 测试2：FAQ查询
    print("\n" + "=" * 50)
    print("测试2：退货咨询")
    print(f"模型回复: {run_agent('我买了手机不满意，可以退货吗？')}")

    # 测试3：不需要工具
    print("\n" + "=" * 50)
    print("测试3：闲聊")
    print(f"模型回复: {run_agent('你好，请问你是谁？')}")

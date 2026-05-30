from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pprint import pprint


# 用 langchain-openai 对接 DeepSeek
llm = ChatOpenAI(
    api_key="sk-c576413004a44dfeb327d8431b612bcb",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
)


@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气，用户需要提供地点名称。参数为中文拼音。"""
    # 模拟返回天气数据
    return "24摄氏度"


llm_with_tools = llm.bind_tools([get_weather])

# 第一轮：用户提问，模型决定调用哪个工具
messages = [HumanMessage(content="杭州的天气怎么样？")]
result = llm_with_tools.invoke(messages)
# print("=" * 50)
# pprint(result.model_dump())
# print("=" * 50)

# 第二轮：执行工具并将结果返回给模型
tool_call = result.tool_calls[0]
messages.append(result)
messages.append(ToolMessage(content="24摄氏度", tool_call_id=tool_call["id"]))
result = llm_with_tools.invoke(messages)
pprint(result.model_dump())

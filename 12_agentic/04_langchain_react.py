from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
)


# ============================================================
# 真正执行的函数（与 01 相同，使用真实天气 API）
# ============================================================
from tianqi import get_weather as _get_weather
import os


@tool
def get_weather(city_name: str) -> str:
    """获取指定城市的天气信息，用户需要提供城市名称, 如大连或大连市"""
    return _get_weather(city_name)


@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
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


tools = [get_weather, get_current_time, search_faq]

# ============================================================
# 一行创建 ReAct Agent，LangGraph 自动处理 Think → Act → Observe 循环
# ============================================================
agent = create_agent(model=llm, tools=tools)


if __name__ == "__main__":
    # 测试1：单工具调用
    print("=" * 50)
    print("测试1：查询天气")
    result = agent.invoke({"messages": [("user", "杭州的天气怎么样？")]})
    print(f"模型回复: {result['messages'][-1].content}")

    # 测试2：FAQ查询
    print("\n" + "=" * 50)
    print("测试2：退货咨询")
    result = agent.invoke({"messages": [("user", "我买了手机不满意，可以退货吗？")]})
    print(f"模型回复: {result['messages'][-1].content}")

    # 测试3：不需要工具
    print("\n" + "=" * 50)
    print("测试3：闲聊")
    result = agent.invoke({"messages": [("user", "你好，请问你是谁？")]})
    print(f"模型回复: {result['messages'][-1].content}")

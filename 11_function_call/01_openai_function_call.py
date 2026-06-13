from openai import OpenAI
from pprint import pprint
import os

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
    base_url="https://api.deepseek.com",
)


def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-v4-flash", messages=messages, tools=tools
    )
    return response.choices[0].message


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气，用户需要提供地点名称。参数为中文拼音。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市，例如：beijing",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

messages = [{"role": "user", "content": "杭州的天气怎么样？"}]
result = send_messages(messages)
pprint(result.model_dump())
# {'annotations': None,
#  'audio': None,
#  'content': '',
#  'function_call': None,
#  'reasoning_content': '用户想知道杭州的天气情况。我需要调用get_weather工具，地点参数应该是"hangzhou"。',
#  'refusal': None,
#  'role': 'assistant',
#  'tool_calls': [{'function': {'arguments': '{"location": "hangzhou"}',
#                               'name': 'get_weather'},
#                  'id': 'call_00_geWJXIKErzvHhZjUyVDZ5422',
#                  'index': 0,
#                  'type': 'function'}]}

# print(f"User>\t {messages[0]['content']}")

messages.append(result)
tool = result.tool_calls[0]

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24摄氏度"})
result = send_messages(messages)
pprint(result.model_dump())

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),  # 替换为你的DeepSeek API Key
    base_url="https://api.deepseek.com",  # DeepSeek API端点
)

response = client.chat.completions.create(
    model="deepseek-chat",  # DeepSeek模型
    messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
)

print(response.choices[0].message.content)

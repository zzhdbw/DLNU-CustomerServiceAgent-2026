from openai import OpenAI

client = OpenAI(
    api_key="sk-23f015edb9a94a7f83f3c1a85753e976",  # 替换为你的DeepSeek API Key
    base_url="https://api.deepseek.com",  # DeepSeek API端点
)

response = client.chat.completions.create(
    model="deepseek-chat",  # DeepSeek模型
    messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
)

print(response.choices[0].message.content)

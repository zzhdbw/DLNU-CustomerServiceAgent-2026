from openai import OpenAI

client = OpenAI(
    api_key="sk-23f015edb9a94a7f83f3c1a85753e976",
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

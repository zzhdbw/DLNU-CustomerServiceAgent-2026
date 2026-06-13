from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),  # 替换为你的DeepSeek API Key
    base_url="https://api.deepseek.com",  # DeepSeek API端点
)

messages = []

while True:
    user_input = input("请输入问题 (输入 'quit' 退出): ")
    if user_input.lower() == 'quit':
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True,
    )

    assistant_message = ""
    print("AI: ", end="")
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            assistant_message += content
    print()
    messages.append({"role": "assistant", "content": assistant_message})

import os
from pprint import pprint

from naive_RAG_01_make_embedding import emb_text
from openai import OpenAI
from pymilvus import MilvusClient

SYSTEM_PROMPT = """
    你是一个手机产品智能客服助手，请根据提供的上下文片段回答用户关于手机的问题。
    """
USER_PROMPT = """
请根据 <context> 标签中的资料回答 <question> 标签中的问题。
<context>
{}
</context>
<question>
{}
</question>

请用中文回答问题
"""

if __name__ == "__main__":
    db_path = "./db_files/milvus_demo.db"
    collection_name = "my_rag_collection"
    question = "小米14的屏幕参数是什么？"

    # 创建向量数据库连接
    milvus_client = MilvusClient(uri=db_path)

    # 搜索与问题最相似的向量
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    pprint(search_res)

    # 组装Prompt
    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])

    # Call DeepSeek API
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek API Key"),
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context, question)},
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

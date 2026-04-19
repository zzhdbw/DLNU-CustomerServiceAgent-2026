from pymilvus import MilvusClient
from naive_RAG_01_make_embedding import emb_text
from pprint import pprint
from openai import OpenAI

SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
USER_PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
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
    question = "How is data stored in milvus?"

    # 创建向量数据库连接
    milvus_client = MilvusClient(uri=db_path)

    # 搜索与问题最相似的向量
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(question)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    pprint(search_res)

    # 组装Prompt
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    # Call DeepSeek API
    client = OpenAI(
        api_key="sk-4e1697139c74434290d1348daa8013d6",
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

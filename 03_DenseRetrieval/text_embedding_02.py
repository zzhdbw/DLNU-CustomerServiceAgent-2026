from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./models/BAAI/bge-m3")

queries = ["什么是向量数据库？", "如何训练 embedding 模型？"]
docs = [
    "向量数据库用于存储和检索高维向量，常见于语义搜索和 RAG。",
    "Embedding 模型通常通过对比学习训练，使语义相近文本更接近。",
    "PostgreSQL 是一个关系型数据库。",
]

q_emb = model.encode_query(queries, normalize_embeddings=True)
d_emb = model.encode_document(docs, normalize_embeddings=True)
print(q_emb)
print(q_emb.shape)
print(d_emb)
print(d_emb.shape)

# 已归一化后，点积就等价于 cosine similarity
import pandas as pd
from tabulate import tabulate

scores = q_emb @ d_emb.T
df_scores = pd.DataFrame(scores, index=queries, columns=docs)
print(tabulate(df_scores, headers='keys', tablefmt='grid', floatfmt='.4f'))
#                     向量数据库用于存储和检索高维向量，常见于语义搜索和 RAG。  Embedding 模型通常通过对比学习训练，使语义相近文本更接近。  PostgreSQL 是一个关系型数据库。
# 什么是向量数据库？                                 0.776141                            0.402241               0.588849
# 如何训练 embedding 模型？                        0.385997                            0.695839               0.398642

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("./models/BAAI/bge-m3")

words = ["国王", "男人", "女人", "皇后"]

embeddings = model.encode(words, normalize_embeddings=True)

word_to_idx = {word: i for i, word in enumerate(words)}
print(word_to_idx)


def cosine_similarity(a, b):
    return np.dot(a, b)


print("=" * 60)
print("词向量类比关系示例：国王 - 男人 + 女人 = 皇后")
print("=" * 60)

vec_king = embeddings[word_to_idx["国王"]]
vec_man = embeddings[word_to_idx["男人"]]
vec_woman = embeddings[word_to_idx["女人"]]
vec_queen = embeddings[word_to_idx["皇后"]]

result_vec = vec_king - vec_man + vec_woman

print("\n【向量计算】国王 - 男人 + 女人")
print(f"  结果向量与各词的相似度：")

similarities = []
for word in words:
    sim = cosine_similarity(result_vec, embeddings[word_to_idx[word]])
    similarities.append((word, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
for word, sim in similarities:
    print(f"  {word}: {sim:.4f}")

print("\n【验证】皇后与结果的相似度：", cosine_similarity(result_vec, vec_queen))

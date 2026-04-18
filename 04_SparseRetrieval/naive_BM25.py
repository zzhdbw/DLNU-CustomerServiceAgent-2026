import math
from collections import Counter
from pprint import pprint


def idf(word: str, count_list: list[Counter]) -> float:
    """计算逆文档频率（Inverse Document Frequency）：衡量该词在整个语料库中的重要程度"""
    n_contain = sum([1 for count in count_list if word in count])
    return math.log((len(count_list) - n_contain + 0.5) / (n_contain + 0.5) + 1)


def bm25(
    word: str,
    count: Counter,
    count_list: list[Counter],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """计算 BM25 分数：一种用于信息检索的排名函数，考虑词频和文档长度归一化"""
    f = count[word]
    dl = sum(count.values())
    word_idf = idf(word, count_list)
    return word_idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl)))


if __name__ == "__main__":
    corpus = [
        'this is the first document',
        'this is the second second document',
        'and the third one',
    ]
    words_list = [doc.split(' ') for doc in corpus]
    pprint(words_list)

    count_list = [Counter(words_list[i]) for i in range(len(words_list))]
    pprint(count_list)

    doc_len_list = [sum(count.values()) for count in count_list]
    avgdl = sum(doc_len_list) / len(doc_len_list)
    print(f"平均文档长度 avgdl = {round(avgdl, 5)}")

    for index, count in enumerate(count_list):
        print(f"第 {index + 1} 个文档 BM25 统计信息")
        for word in count:
            print(f"{word}: {bm25(word, count, count_list, avgdl):.5f}")

        print("=" * 50)

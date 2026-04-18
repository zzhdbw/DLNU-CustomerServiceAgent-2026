import math
from collections import Counter
from pprint import pprint


def tf(word: str, count: Counter) -> float:
    """计算词频（Term Frequency）：该词在当前文档中出现次数占文档总词数的比例"""
    return count[word] / sum(count.values())


def idf(word: str, count_list: list[Counter]) -> float:
    """计算逆文档频率（Inverse Document Frequency）：衡量该词在整个语料库中的普遍程度"""
    n_contain = sum(
        [1 for count in count_list if word in count]
    )  # 该词出现在几个文档中
    return math.log(len(count_list) / (1 + n_contain))  # 加1是为了避免除零错误


def tf_idf(word: str, count: Counter, count_list: list[Counter]) -> float:
    """计算TF-IDF值：结合词频和逆文档频率，评估词语在特定文档中的重要程度"""
    return tf(word, count) * idf(word, count_list)


if __name__ == "__main__":
    corpus = [
        'this is the first document',
        'this is the second second document',
        'and the third one',
    ]
    # 分词
    words_list = [doc.split(' ') for doc in corpus]
    pprint(words_list)

    # 统计每个文档中每个词的出现次数
    count_list = [Counter(words_list[i]) for i in range(len(words_list))]
    pprint(count_list)

    for index, count in enumerate(count_list):
        print(f"第 {index + 1} 个文档 TF-IDF 统计信息")
        for word in count:
            print(f"{word}: {tf_idf(word, count, count_list):.5f}")
        print("=" * 50)

import math
from collections import Counter


class BM25Retriever:
    def __init__(self, documents, use_jieba=True, stopwords=None, k1=1.5, b=0.75):
        """
        documents: list[dict]
            例如:
            [
                {"id": 1, "title": "机器学习入门", "text": "机器学习是人工智能的重要分支"},
                ...
            ]

        use_jieba: 是否使用 jieba 分词
        stopwords: 停用词集合
        k1, b: BM25 参数
        """
        self.documents = documents
        self.use_jieba = use_jieba
        self.stopwords = stopwords or set()
        self.k1 = k1
        self.b = b

        self.tokenized_docs = []
        self.doc_lens = []
        self.avgdl = 0.0
        self.doc_freqs = {}  # df(term)
        self.idf = {}  # idf(term)
        self.term_freqs = []  # 每篇文档的词频 Counter

        self._build_index()

    def tokenize(self, text):
        if self.use_jieba:
            import jieba

            # 搜索场景下，用搜索引擎模式更像检索系统
            words = list(jieba.cut_for_search(text))
        else:
            # 不使用 jieba 时，要求文本已经按空格切好
            words = text.split()

        result = []
        for w in words:
            w = w.strip().lower()
            if not w:
                continue
            if w in self.stopwords:
                continue
            result.append(w)
        return result

    def _build_index(self):
        # 1) 分词
        self.tokenized_docs = [self.tokenize(doc["text"]) for doc in self.documents]

        # 2) 文档长度
        self.doc_lens = [len(tokens) for tokens in self.tokenized_docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

        # 3) 每篇文档词频
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized_docs]

        # 4) 计算 df
        self.doc_freqs = {}
        for tokens in self.tokenized_docs:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # 5) 计算 idf
        # 常见 BM25 写法：
        # idf = log((N - df + 0.5) / (df + 0.5) + 1)
        n_docs = len(self.tokenized_docs)
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def _score_one(self, query_terms, doc_index):
        """
        计算单篇文档对 query 的 BM25 分数
        """
        score = 0.0
        tf_counter = self.term_freqs[doc_index]
        dl = self.doc_lens[doc_index]

        for term in query_terms:
            if term not in tf_counter:
                continue

            tf = tf_counter[term]
            idf = self.idf.get(term, 0.0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=3):
        query_terms = self.tokenize(query)

        results = []
        for i, doc in enumerate(self.documents):
            score = self._score_one(query_terms, i)
            results.append({"score": score, "doc": doc})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def explain_query(self, query):
        """
        打印 query 中每个词的 idf，方便调试
        """
        query_terms = self.tokenize(query)
        print("查询分词:", query_terms)
        print("query term 的 IDF:")
        for term in query_terms:
            print(f"  {term:<12} {self.idf.get(term, 0.0):.4f}")

    def explain_doc_score(self, query, doc_index):
        """
        解释某个 query 对某篇文档的打分过程
        """
        query_terms = self.tokenize(query)
        tf_counter = self.term_freqs[doc_index]
        dl = self.doc_lens[doc_index]

        print(f"文档: {self.documents[doc_index]['title']}")
        print(f"文档长度 dl={dl}, 平均长度 avgdl={self.avgdl:.4f}")
        print()

        total_score = 0.0
        for term in query_terms:
            tf = tf_counter.get(term, 0)
            if tf == 0:
                print(f"{term:<12} tf=0 -> contribution=0")
                continue

            idf = self.idf.get(term, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            part = idf * (numerator / denominator)

            total_score += part
            print(f"{term:<12} tf={tf:<3} idf={idf:.4f} " f"part={part:.4f}")

        print(f"\n总分 score = {total_score:.4f}")


if __name__ == "__main__":
    documents = [
        {
            "id": 1,
            "title": "机器学习入门",
            "text": "机器学习是人工智能的重要分支，常见方法包括监督学习和无监督学习",
        },
        {
            "id": 2,
            "title": "自然语言处理简介",
            "text": "自然语言处理关注文本分析、分词、词向量、文本分类和信息抽取",
        },
        {
            "id": 3,
            "title": "深度学习基础",
            "text": "深度学习是机器学习的一个重要方向，常用于图像识别、语音识别和文本建模",
        },
        {
            "id": 4,
            "title": "搜索引擎技术",
            "text": "搜索引擎通常包括分词、倒排索引、相关性排序和检索召回等模块",
        },
        {
            "id": 5,
            "title": "中文分词实践",
            "text": "中文文本处理中，分词是基础步骤，jieba 常用于中文分词和搜索场景",
        },
    ]

    stopwords = {"是", "的", "和", "中", "一个", "通常", "包括", "用于", "常见"}

    retriever = BM25Retriever(
        documents=documents, use_jieba=True, stopwords=stopwords, k1=1.5, b=0.75
    )

    query = "中文文本分词检索"

    retriever.explain_query(query)

    print("\n检索结果：")
    results = retriever.search(query, top_k=3)
    for rank, item in enumerate(results, start=1):
        doc = item["doc"]
        print(
            f"{rank}. score={item['score']:.4f} | id={doc['id']} | title={doc['title']}"
        )
        print(f"   text={doc['text']}")

    print("\n\n===== 打分解释示例 =====")
    retriever.explain_doc_score(query, doc_index=4)  # 看第5篇文档

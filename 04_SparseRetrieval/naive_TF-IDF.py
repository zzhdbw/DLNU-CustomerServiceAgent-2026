import math
from collections import Counter


class TfidfRetriever:
    def __init__(self, documents, use_jieba=True, stopwords=None):
        """
        documents: list[dict], 例如:
            [
                {"id": 1, "title": "机器学习入门", "text": "机器学习是人工智能的重要分支"},
                ...
            ]
        use_jieba: 是否使用 jieba 分词
        stopwords: 停用词集合
        """
        self.documents = documents
        self.use_jieba = use_jieba
        self.stopwords = stopwords or set()

        self.tokenized_docs = []
        self.vocab = []
        self.df = {}
        self.idf = {}
        self.doc_vectors = []
        self.doc_norms = []

        self._build_index()

    def tokenize(self, text):
        if self.use_jieba:
            import jieba

            # 搜索场景可以用 cut_for_search，提高召回
            words = list(jieba.cut_for_search(text))
        else:
            # 不用 jieba 时，要求文本已经空格分词
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
        # 1. 文档分词
        self.tokenized_docs = [self.tokenize(doc["text"]) for doc in self.documents]

        # 2. 词表
        self.vocab = sorted(set(word for doc in self.tokenized_docs for word in doc))

        # 3. 计算 DF
        self.df = {}
        for word in self.vocab:
            self.df[word] = sum(1 for doc in self.tokenized_docs if word in doc)

        # 4. 计算 IDF
        n_docs = len(self.tokenized_docs)
        self.idf = {}
        for word in self.vocab:
            self.idf[word] = math.log((1 + n_docs) / (1 + self.df[word])) + 1

        # 5. 计算每篇文档的 TF-IDF 向量
        self.doc_vectors = []
        self.doc_norms = []

        for tokens in self.tokenized_docs:
            vec = self._compute_tfidf_vector(tokens)
            self.doc_vectors.append(vec)
            self.doc_norms.append(self._vector_norm(vec))

    def _compute_tfidf_vector(self, tokens):
        """
        返回稀疏向量 dict: {term: tfidf}
        """
        total = len(tokens)
        counter = Counter(tokens)

        vec = {}
        if total == 0:
            return vec

        for term, cnt in counter.items():
            tf = cnt / total
            idf = self.idf.get(term, 0.0)
            vec[term] = tf * idf
        return vec

    def _vector_norm(self, vec):
        return math.sqrt(sum(v * v for v in vec.values()))

    def _cosine_similarity(self, vec1, norm1, vec2, norm2):
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 遍历更小的向量，减少计算量
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1

        dot = 0.0
        for term, value in vec1.items():
            dot += value * vec2.get(term, 0.0)

        return dot / (norm1 * norm2)

    def search(self, query, top_k=3):
        query_tokens = self.tokenize(query)
        query_vec = self._compute_tfidf_vector(query_tokens)
        query_norm = self._vector_norm(query_vec)

        results = []
        for i, doc in enumerate(self.documents):
            score = self._cosine_similarity(
                query_vec, query_norm, self.doc_vectors[i], self.doc_norms[i]
            )
            results.append({"score": score, "doc": doc})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def explain_query(self, query):
        """
        打印 query 的 TF-IDF 权重，方便调试
        """
        tokens = self.tokenize(query)
        vec = self._compute_tfidf_vector(tokens)
        sorted_items = sorted(vec.items(), key=lambda x: x[1], reverse=True)

        print(f"查询分词: {tokens}")
        print("查询 TF-IDF 权重:")
        for term, score in sorted_items:
            print(f"  {term:<10} {score:.4f}")


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

    stopwords = {"是", "的", "和", "中", "一个", "通常", "包括", "用于", "常用"}

    retriever = TfidfRetriever(documents=documents, use_jieba=True, stopwords=stopwords)

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

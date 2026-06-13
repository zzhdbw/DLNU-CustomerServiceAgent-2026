# DLNU 客服 Agent

从 PDF 文档解析到智能客服 Agent 的完整 RAG 技术栈教学项目。

## 快速开始

```bash
# 1. 安装 uv（Python 包管理器）
pip install uv

# 2. 创建虚拟环境
uv venv --python=3.10
source .venv/bin/activate

# 3. 安装依赖
uv pip install -r requirements.txt

# 4. 配置 API Key（复制并编辑 .env）
cp .env.example .env
```

## 环境变量

请在 `.env` 中配置以下 API Key（项目中硬编码的 Key 已全部替换为环境变量）：

| 变量名 | 用途 | 获取方式 |
|--------|------|----------|
| `DEEPSEEK_API_KEY` | DeepSeek 大模型调用 | [deepseek.com](https://platform.deepseek.com) |
| `JINA_API_KEY` | Jina Embeddings 文本向量化 | [jina.ai](https://jina.ai/embeddings) |
| `AMAP_API_KEY` | 高德天气查询 | [lbs.amap.com](https://lbs.amap.com) |
| `API_TOKEN` | Mineru PDF 解析 | [mineru.net](https://mineru.net) |

---

## 模块导航

| # | 模块 | 核心内容 |
|---|------|----------|
| 01 | **DocumentParse** | PDF 解析为 Markdown（Mineru API / PyPDF2） |
| 02 | **DocumentSplit** | 文本切分（固定长度 / 递归 / 语义切分） |
| 03 | **DenseRetrieval** | 稠密向量检索（SentenceTransformer / BGE-M3） |
| 04 | **SparseRetrieval** | 稀疏检索（TF-IDF / BM25 原理实现） |
| 05 | **VectorDB** | 向量数据库（Milvus Lite 基础操作） |
| 06 | **CallLLM** | 大模型 API 调用（DeepSeek，流式 / 多轮） |
| 07 | **Gradio** | 可视化界面（Gradio ChatInterface） |
| 08 | **Naive RAG** | 手写 RAG 完整流程（嵌入 + 检索 + 界面） |
| 09 | **LangChain RAG** | 基于 LangChain 的 RAG（Milvus + Jina） |
| 10 | **LangGraph RAG** | 基于 LangGraph 的 RAG 流程编排 |
| 11 | **Function Call** | 函数调用（OpenAI / LangChain 两种方式） |
| 12 | **Agentic** | 智能体（OpenAI / LangChain / ReAct Agent） |
| 13 | **Agentic RAG** | RAG + 工具调用的 Agent 客服系统 |

---

### 01 — DocumentParse

PDF 文档解析为可处理的 Markdown 文本。

```bash
cd 01_DocumentParse
python pdf2markdown_01.py    # Mineru API 提交任务
python pdf2markdown_02.py    # 下载解析结果
python pdf2txt_py.py         # PyPDF2 本地解析
```

### 02 — DocumentSplit

将长文本切分为适合检索的 Chunks。

```bash
cd 02_DocumentSplit
python split_text_01.py      # 固定长度切分
python split_text_02.py      # 递归分隔符切分
python split_text_03.py      # 语义切分（需 spacy 模型）
```

### 03 — DenseRetrieval

使用 SentenceTransformer / BGE-M3 生成稠密向量，计算语义相似度。

### 04 — SparseRetrieval

手动实现 TF-IDF 和 BM25 稀疏检索算法，理解检索原理。

### 05 — VectorDB

Milvus Lite 向量数据库基础操作：创建集合、索引、插入、检索。

### 06 — CallLLM

调用 DeepSeek API 的四种方式：流式 / 非流式 / 单轮 / 多轮。

### 07 — Gradio

基于 Gradio 构建 LLM 交互界面，支持流式输出。

```bash
cd 07_Gradio
python 03_LLM_multi_round.py          # 多轮对话
python 04_LLM_multi_round_stream.py   # 多轮 + 流式
```

### 08 — Naive RAG

从零手写 RAG 全流程：调用 Jina API 生成 Embedding → Milvus 检索 → LLM 生成回答。

```bash
cd 08_naive_RAG
python naive_RAG_01_make_embedding.py   # 构建向量库
python naive_RAG_02_retrieval.py        # 检索 + 回答
python naive_RAG_03_gradio.py           # Gradio 界面
```

### 09 — LangChain RAG

使用 LangChain 框架 + Milvus + Jina Embeddings 构建手机客服 RAG 系统。

```bash
cd 09_langchain_RAG
python 01_build_vectorstore.py          # 构建向量库
python 02_retrieval.py                  # 检索 + 回答
python 03_retrieval_with_gradio.py      # Gradio 界面（端口 7863）
```

### 10 — LangGraph RAG

使用 LangGraph 的 StateGraph 编排 RAG 流程，支持条件分支和循环。

### 11 — Function Call

大模型函数调用能力：用 OpenAI SDK 和 LangChain 两种方式定义工具。

```bash
cd 11_function_call
python 01_openai_function_call.py       # OpenAI SDK 方式
python 02_langchain_function_call.py    # LangChain 方式
```

### 12 — Agentic

多种智能体实现方案：从手写 ReAct 循环到 LangChain/LangGraph Agent。

```bash
cd 12_agentic
python 01_openai_agentic.py             # OpenAI SDK 手写 Agent
python 02_langchain_agentic.py          # LangChain Agent
python 03_openai_react.py               # OpenAI + ReAct 循环
python 04_langchain_react.py            # LangChain ReAct Agent
```

内置工具：天气查询（高德 API）、FAQ 搜索、系统时间。

### 13 — Agentic RAG（综合项目）

将 RAG 与 Agent 结合：手机产品资料检索 + 天气查询 + 时间查询，基于 LangChain ReAct Agent + Gradio。

```bash
cd 13_agentic_RAG
python 01_build_vectorstore.py          # 构建手机文档向量库
python 02_agentic_rag_gradio.py          # 启动 Agent 界面（端口 7865）
```

---

## 项目路线图

```
PDF 解析 → 文本切分 → Embedding → 向量库 → 检索 → LLM 生成 → 工具调用 → Agent
  (01)      (02)     (03/04)     (05)   (08/09)   (06/07)    (11/12)   (13)
```

从底层原理到框架应用，逐步构建完整的智能客服 Agent 系统。

# LangChain RAG 手机智能客服

基于 **LangChain + Milvus + DeepSeek + Gradio** 的检索增强生成（RAG）智能客服系统。支持手机产品售前咨询与售后问题解答，提供可视化的 Gradio 对话界面。

## 项目结构

```
langchain_RAG/
├── data/phone_docs/zh/       # 手机知识库（Markdown）
│   ├── specs/                # 手机规格参数（小米、华为、红米等）
│   ├── pre_sale/             # 售前资料（比价、功能、库存等）
│   └── after_sale/           # 售后资料（退换货、保修、网络等）
├── db_files/                 # Milvus 向量数据库文件
├── 01_build_vectorstore.py   # 构建向量数据库
├── 02_retrieval.py           # RAG 检索问答（命令行）
└── 03_retrieval_with_gradio.py  # RAG 检索问答（Gradio 界面）
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 向量库 | Milvus Lite（嵌入式，无需独立部署） |
| 文本嵌入 | Jina Embeddings v5（768 维） |
| 文本切分 | MarkdownHeaderTextSplitter（按 H1/H2/H3 层级切分） |
| LLM | DeepSeek Chat（通过 OpenAI 兼容接口调用） |
| 框架 | LangChain（LCEL 表达式语言） |
| 界面 | Gradio ChatInterface（流式输出） |

## 快速开始

### 1. 创建 conda 环境并安装依赖

```bash
conda create -n langchain_RAG python=3.12 -y
conda activate langchain_RAG
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. 配置环境变量

在 `01_build_vectorstore.py` / `02_retrieval.py` / `03_retrieval_with_gradio.py` 中已内置 Jina API Key 和 DeepSeek API Key，或替换为自己的 Key：

- **Jina Embeddings**: https://jina.ai/embeddings/
- **DeepSeek**: https://platform.deepseek.com/

### 3. 构建向量数据库

```bash
python 01_build_vectorstore.py
```

流程：加载 `data/phone_docs/zh/` 下所有 Markdown 文件 → 按标题层级切分 → 生成嵌入向量 → 存入 Milvus。

### 4. 运行问答

**命令行模式：**
```bash
python 02_retrieval.py
```

**Gradio 可视化界面：**
```bash
python 03_retrieval_with_gradio.py
```

访问 `http://localhost:7863` 使用对话界面。

## Gradio 界面功能

- 多轮对话记忆（保持上下文）
- RAG 检索开关（可对比有无检索的回答效果）
- TopK 检索数量调节（1-10 条）
- 检索结果展示（来源文件 + 相似度分数）
- 预设示例问题
- 流式输出回答

## 数据说明

知识库包含三类手机相关文档：

- **specs/**: 各型号详细规格（小米 13/14/15 Ultra、华为 Mate60 Pro/P60/Nova12、红米 K70）
- **pre_sale/**: 售前信息（机型对比、功能 Q&A、价格促销、物流、库存）
- **after_sale/**: 售后信息（退换货、维修保修、数据账号、网络信号、隐私安全、常见问题）
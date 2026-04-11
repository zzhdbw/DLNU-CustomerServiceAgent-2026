# DLNU 客服 Agent

客服 Agent 文档处理模块，用于将 PDF 文档转换为可用的文本数据。

## 环境

```bash
uv venv --python=3.10
source .venv/bin/activate
uv --cache-dir ~/.uv/cache  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -r requirements.txt
```

需要在 `.env` 文件中配置 Mineru API Token：
```
API_TOKEN=your_token_here
```

## 模块

### 01_DocumentParse - PDF 转 Markdown

使用 [Mineru](https://mineru.net/) 将 PDF 解析为 Markdown 格式。

| 文件 | 说明 |
|------|------|
| `pdf2markdown_01.py` | 提交解析任务，获取 task_id |
| `pdf2markdown_02.py` | 根据 task_id 下载解析结果（ZIP） |

### 02_DocumentSplit - 文本切分

将长文本切分为适合处理的 chunks。

| 文件 | 说明 |
|------|------|
| `split_text_01.py` | 按固定长度切分 |
| `split_text_02.py` | 递归按分隔符切分，保留句子完整性 |
| `split_text_03.py` | 语义切分（基于 spaCy 句子边界） |

## 使用

### 1. 解析 PDF

```bash
cd 01_DocumentParse

# 步骤1：提交任务
python pdf2markdown_01.py
# 输出 task_id

# 步骤2：下载结果（替换为实际 task_id）
# 修改 pdf2markdown_02.py 中的 task_id
python pdf2markdown_02.py
```

### 2. 切分文本

```bash
cd 02_DocumentSplit

# 按长度切分
python split_text_01.py

# 按分隔符切分
python split_text_02.py

# 语义切分（需安装 spacy 模型）
python -m spacy download en_core_web_sm
python split_text_03.py
```

## 数据目录

```
01_DocumentParse/data/
├── raw/                    # 原始 PDF
├── processed/             # 解析结果
└── processed_zip/         # 下载的 ZIP 文件

02_DocumentSplit/data/
└── *.md                   # 待处理的 Markdown 文件
```

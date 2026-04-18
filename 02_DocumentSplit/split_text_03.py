# https://spacy.io/
import os
import spacy


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "atten_is_all_you_need.md")
CHUNK_SIZE = 512


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text_by_semantic(
    text: str, chunk_size: int = 500, chunk_overlap: int = 20
) -> list[str]:
    """核心分割逻辑"""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # 动态合并句子
    current_chunk = []
    current_length = 0
    chunks = []

    for sent in sentences:
        sent_length = len(sent)

        # 判断是否超过阈值
        if current_length + sent_length > chunk_size:
            if current_chunk:
                # 中文用空字符串连接
                chunks.append("".join(current_chunk))

                # 精确计算重叠字符数
                overlap_buffer = []
                overlap_length = 0
                # 逆向遍历寻找重叠边界
                for s in reversed(current_chunk):
                    if overlap_length + len(s) > chunk_overlap:
                        break
                    overlap_buffer.append(s)
                    overlap_length += len(s)
                # 恢复原始顺序
                current_chunk = list(reversed(overlap_buffer))
                current_length = overlap_length

        current_chunk.append(sent)
        current_length += sent_length

    # 处理剩余内容
    if current_chunk:
        chunks.append("".join(current_chunk))
    return chunks


if __name__ == "__main__":
    text = read_file(DATA_PATH)
    nlp = spacy.load("zh_core_web_sm")

    chunks = split_text_by_semantic(
        text,
        chunk_size=500,
        chunk_overlap=0,
    )

    print(f"文件总长度: {len(text)} 字符")
    print(f"切分块数: {len(chunks)}")
    print("-" * 60)

    for i, chunk in enumerate(chunks, 1):
        print(f"===== Chunk {i} (长度: {len(chunk)}) ========================")

        print(chunk)
        print("==================================================")
        print()

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "atten_is_all_you_need.md")
CHUNK_SIZE = 1000


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text_recursive(
    text: str,
    chunk_size=500,
    chunk_overlap=20,
    separators=["\n", " ", "。", "，", "！", "？"],
) -> list[str]:
    """递归分割文本"""
    # 1. 如果文本长度小于chunk_size，直接返回
    if len(text) <= chunk_size:
        return [text]
    # 2. 递归尝试不同的分隔符
    for separator in separators:
        if separator in text:
            chunks = []
            splits = text.split(separator)
            current_chunk = []
            current_length = 0
            # 3. 根据chunk_size组合文本块
            for split in splits:
                if current_length + len(split) > chunk_size:
                    # 创建新chunk
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    current_chunk = [split]
                    current_length = len(split)
                else:
                    current_chunk.append(split)
                    current_length += len(split)
            # 4. 处理重叠
            final_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # 添加前一个chunk的结尾
                    overlap_start = max(0, len(chunks[i - 1]) - chunk_overlap)
                    chunk = chunks[i - 1][overlap_start:] + chunk
                final_chunks.append(chunk)

            return final_chunks


if __name__ == "__main__":
    text = read_file(DATA_PATH)
    chunks = split_text_recursive(
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

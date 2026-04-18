import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "atten_is_all_you_need.md")
CHUNK_SIZE = 1000


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_by_length(text, chunk_size):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


if __name__ == "__main__":
    text = read_file(DATA_PATH)
    chunks = split_by_length(text, CHUNK_SIZE)

    print(f"文件总长度: {len(text)} 字符")
    print(f"切分块数: {len(chunks)}")
    print("-" * 50)

    for i, chunk in enumerate(chunks, 1):
        print(f"===== Chunk {i} (长度: {len(chunk)}) ========================")
        print(chunk)
        print("==================================================")
        print()

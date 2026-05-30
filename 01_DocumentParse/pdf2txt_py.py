import PyPDF2

pdf_reader = PyPDF2.PdfReader("data/raw/NIPS-2017-attention-is-all-you-need-Paper.pdf")
text = ""

for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    text += page.extract_text() + "\n"
print(text)

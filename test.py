import fitz  # PyMuPDF
from sympy import sympify

# Open the PDF
pdf_document = "file.pdf"
doc = fitz.open(pdf_document)

# Extract content from each page
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()  # This will include text, but not retain LaTeX-style math
    print(f"Page {page_num + 1}:\n{text}")

# Extracting symbols (using regex or SymPy parsing if required)
# This approach works when math expressions are well defined
math_data = sympify(text)
print(math_data)

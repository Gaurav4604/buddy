import fitz  # PyMuPDF
from pdf2image import convert_from_path
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pix2text import Pix2Text, merge_line_texts
import pytesseract
from ollama import Client


# Path to the PDF file
pdf_path = "files/automata.pdf"

p2t = Pix2Text()


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
base_url = OLLAMA_URL
ollama = Client(host=base_url)


output_dir = "raw"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Open the PDF
doc = fitz.open(pdf_path)
total_pages = doc.page_count


system_prompt = f"convert the data to tex file format only, display nothing else, other than the required tex. ensure that it has tags such as begin, section, subsection and has usepackage amsmath"

# Convert each page to image and extract text
for page_num in range(total_pages):
    # Convert the PDF page to image
    pages = convert_from_path(
        pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300
    )

    # Run OCR on the image to extract text
    p2t_text = p2t.recognize(img=pages[0])
    tess_text = pytesseract.image_to_string(pages[0])

    prompt_1 = f"""I have a document page extracted out, it has an OCR based extraction: {tess_text}, this does not have valid math symbols"""

    prompt_2 = (
        f"""this is a math based extraction latex format extraction: {p2t_text}"""
    )

    prompt_3 = """merge it and convert it to proper latex document"""
    res = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt_1,
            },
            {
                "role": "user",
                "content": prompt_2,
            },
            {
                "role": "user",
                "content": prompt_3,
                "options": {system_prompt: system_prompt},
            },
        ],
    )
    output = res["message"]["content"]

    text_file_name = os.path.join(output_dir, f"{page_num}.txt")
    with open(text_file_name, "w", encoding="utf-8") as text_file:
        text_file.write(output)

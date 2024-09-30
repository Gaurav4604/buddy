import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import os

# Path to the PDF file
pdf_path = "file.pdf"

# Create directory named 'rough' if it doesn't exist
output_dir = "rough"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the PDF
doc = fitz.open(pdf_path)
total_pages = doc.page_count

# Convert each page to image and extract text
for page_num in range(total_pages):
    # Convert the PDF page to image
    pages = convert_from_path(
        pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300
    )

    # Run OCR on the image to extract text
    for page_image in pages:
        text = pytesseract.image_to_string(page_image)

        # Save extracted text to a .txt file inside the 'rough' folder (1.txt, 2.txt, etc.)
        text_file_name = os.path.join(output_dir, f"{page_num + 1}.txt")
        with open(text_file_name, "w", encoding="utf-8") as text_file:
            text_file.write(text)

print(f"Text from {total_pages} pages has been saved in the 'rough' directory.")

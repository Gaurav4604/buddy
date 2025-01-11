import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import fitz  # PyMuPDF
from pdf2image import convert_from_path
from ultralytics import YOLO
from pix2text import Pix2Text
import pytesseract
from ollama import Client
from utils import process_pdf_with_yolo

model = YOLO("document_semantic_inference.pt")


# Path to the PDF file
pdf_path = "files/automata.pdf"

p2t = Pix2Text()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
base_url = OLLAMA_URL
ollama = Client(host=base_url)

os.makedirs("images", exist_ok=True)

output_dir = "raw"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Open the PDF
doc = fitz.open(pdf_path)
total_pages = doc.page_count

# Inference for Images in Docs
inferences = process_pdf_with_yolo(pdf_path, model)

summary = ""


def validate_text_with_bespoke(text):
    # Use bespoke-minicheck to gracefully verify LaTeX validity, ignoring other invalid content
    res = ollama.chat(
        model="bespoke-minicheck",
        messages=[
            {
                "role": "user",
                "content": f"Check this extraction for valid LaTeX symbols and math. Even if the text outside of LaTeX is invalid, focus on verifying the LaTeX itself: {text}. Give me a Yes/No reply to the same",
            },
        ],
    )
    # If it returns 'yes', proceed with the text
    return "yes" in res["message"]["content"].lower()


def extract_text(image, convert_to_tex=False):
    # Perform OCR and LaTeX extraction
    p2t_text = p2t.recognize(img=image)
    tess_text = pytesseract.image_to_string(image)

    # Assume tess_text is valid and check if p2t_text has valid math
    valid_p2t_math = validate_text_with_bespoke(p2t_text)

    # Create the base prompt using tess_text
    prompt_1 = f"OCR extraction (assumed valid): {tess_text}"

    # Only include p2t_text if it has valid math
    if valid_p2t_math:
        prompt_2 = f"LaTeX extraction (contains valid math): {p2t_text}"
    else:
        prompt_2 = None

    # Determine what the final merge prompt will be
    if convert_to_tex:
        prompt_3 = (
            "Merge and convert to a valid LaTeX document, omitting invalid symbols."
        )
    else:
        prompt_3 = (
            "Merge OCR and LaTeX outputs into a text caption. "
            "If LaTeX contains invalid math, ignore it. Prioritize the valid OCR output."
        )

    # Build the message for the LLM
    messages = [{"role": "user", "content": prompt_1}]
    if prompt_2:
        messages.append({"role": "user", "content": prompt_2})
    messages.append({"role": "user", "content": prompt_3})

    # Call Ollama's LLaMA for the final output generation
    res = ollama.chat(model="llama3.2", messages=messages)
    output = res["message"]["content"]
    return output


def summarize_text(output, summary):
    res = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"This is the current summary for my document: {summary}. This is the new data: {output}",
            },
            {
                "role": "user",
                "content": "Summarize and merge the new data into the existing summary, while maintaining the previous summary, reduce number of words used, so that the summary is not too long. limit the summary to 2 paragraphs. Don't retain any latex formatting or add any system-related info to the summary, just summarize it as simple text.",
            },
        ],
    )

    summary = res["message"]["content"]
    return summary


def embed_image_data(output, image_data, caption_text, used_summary=False):
    surrounding_text = (
        f"This is the previous summary of the document: {caption_text}"
        if used_summary
        else f"This is a surrounding text for image data: {caption_text}"
    )
    merge_instruction = (
        "based on the previous summary"
        if used_summary
        else "using the context from surrounding text"
    )

    res = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"This is the current state of my document: {output}.",
            },
            {
                "role": "user",
                "content": f"This is the image data for the document: {image_data}. This is the surrounding text around the image: {surrounding_text}.",
            },
            {
                "role": "user",
                "content": (
                    f"Add the image data to the existing document {merge_instruction}. "
                    "Ensure that the current document remains intact, and append the new information where necessary. "
                    "Do not replace existing content. Do not add any tags related to *figure* in the merged document, and don't mention any image URLs either."
                ),
                "options": {
                    "system_prompt": (
                        "Convert the data to TeX file format only, display nothing else, other than the required TeX. "
                        "Ensure that it has tags such as \\begin, \\section, \\subsection, and includes \\usepackage{amsmath}."
                    ),
                },
            },
        ],
    )

    result = res["message"]["content"]
    return result


def caption_image(image, caption_path, summary):
    caption = ""
    used_summary = False

    # Generate caption based on valid tess_text, ignoring invalid p2t_text if necessary
    if len(caption_path) > 0:
        caption = extract_text(caption_path, convert_to_tex=False)
        message = (
            f"Explain the image context: {caption}, and use this summary: {summary}"
        )
    else:
        caption = summary
        used_summary = True
        message = f"Based on this summary, explain the image context: {caption}"

    # Call the LLM to generate the final caption for the image
    res = ollama.chat(
        model="minicpm-v",
        messages=[
            {"role": "user", "content": message},
            {"role": "user", "images": [image]},
        ],
    )
    image_caption = res["message"]["content"]
    return {
        "image_info": image_caption,
        "used_summary": used_summary,
        "caption": caption,
    }


# Convert each page to image and extract text
for page_num in range(total_pages):
    # Convert the PDF page to image
    pages = convert_from_path(
        pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=500
    )

    output = extract_text(pages[0], convert_to_tex=True)

    print(f"Processed Raw Page\n{output}\n")

    summary = summarize_text(output, summary)
    print(f"Current Summary\n{summary}\n")

    # Extracting image info from the inferences
    targetInference = inferences[page_num]
    if len(targetInference["results"]) > 0:
        results = targetInference["results"]
        for result in results:
            picture = result["picture"]
            caption = result["caption"]

            caption_result = caption_image(picture, caption, summary)
            print(
                f"Image Info: {caption_result['image_info']}\nCaption: {caption_result['caption']}\nUsed Summary: {caption_result['used_summary']}\n"
            )
            output = embed_image_data(
                output,
                caption_result["image_info"],
                caption_result["caption"],
                caption_result["used_summary"],
            )
            print(f"Current Output after Embedding Image\n{output}\n")

        summary = summarize_text(output, summary)
        print(f"Current Summary\n{summary}\n")

    text_file_name = os.path.join(output_dir, f"{page_num}.txt")
    with open(text_file_name, "w", encoding="utf-8") as text_file:
        text_file.write(output)

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pix2text import Pix2Text, merge_line_texts
import pytesseract
from ollama import Client

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
base_url = OLLAMA_URL
ollama = Client(host=base_url)

res = ollama.chat(
    model="llava-phi3",
    messages=[
        {
            "role": "user",
            "content": """based on this information, tell me what is happening in the image, and what is the context of the image.:
            
All this behaviour can be understood just by putting your finger on the current state (the one
with the dot) and, upon receipt of an event, following the appropriate transition to the next state.
By reading the diagram, it is easy to see what will happen in every situation.
Consider our previous diagram, with the tap-light in the OFF state. If we receive a tapevent,
the diagram changes to the ON state, as shown in fig. 3.
            """,
            "images": [
                "images/extracted_page1_3.png",
            ],
        },
    ],
)
output = res["message"]["content"]
print(output)

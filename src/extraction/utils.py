import cv2
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from pylatexenc.latex2text import LatexNodes2Text
from pydantic import BaseModel
import pytesseract
import os
from PIL import Image, ImageOps
from doclayout_yolo import YOLOv10
import base64
from transformers.utils import logging
import warnings
import ollama
import shutil

logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)

latex_model = load_model()
latex_processor = load_processor()


model = YOLOv10("inf.pt")

DO_NOT_MERGE = {3, 5, 8}  # 3=image, 5=formula, 8=table

ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")


client = ollama.AsyncClient(host=ollama_url)


class DocNeedsOCR(BaseModel):
    invalid_latex: bool


class DocContent(BaseModel):
    text: str


class ImageDescription(BaseModel):
    description: str


class Formula(BaseModel):
    formula: str


class DetectTableFormula(BaseModel):
    table_or_formula: str


class Table(BaseModel):
    headers: list[str]
    rows: list[list[str]]

    def construct_table(self) -> str:
        """Constructs a table string, from the data on the object"""
        return r"""
<headers>
    {headers}
</headers>
<rows>
        {rows}
</rows>
        """.format(
            headers=str(self.headers),
            rows="\n\t".join(f"<row>{str(row)}</row>" for row in self.rows),
        )


def pad_image_to_aspect_ratio(image_path, output_path, target_aspect_ratio):
    # Open the original image
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Calculate target dimensions
    target_width = original_width
    target_height = int(target_width * target_aspect_ratio)

    if target_height < original_height:
        target_height = original_height
        target_width = int(target_height / target_aspect_ratio)

    # Calculate padding
    padding_left = (target_width - original_width) // 2
    padding_top = (target_height - original_height) // 2
    padding_right = target_width - original_width - padding_left
    padding_bottom = target_height - original_height - padding_top

    # Apply padding
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    padded_img = ImageOps.expand(img, padding, fill="white")

    # Save the padded image
    padded_img.save(output_path)


def convert_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode("utf-8")
        return base64_string


def save_detections(result: list, save_path: str = "result.jpg"):
    # Annotate and save the result
    annotated_frame = result[0].plot(pil=True, line_width=5, font_size=20)
    cv2.imwrite(save_path, annotated_frame)


async def classify_table_formula(image: str) -> str:
    """
    Classifies whether the image is a table or a formula
    Args:
        image (str): the image url of table to be extracted
    Returns:
        str: classification
    """

    async def get_inference(image: str):
        res = await client.chat(
            model="minicpm-v-2",
            messages=[
                {
                    "role": "user",
                    "content": "is this a table? or a formula?",
                    "images": [convert_to_base64(image)],
                }
            ],
            format=DetectTableFormula.model_json_schema(),
            options={"temperature": 0},
            keep_alive=0,
        )
        return res

    try:
        res = await get_inference(image)
    except:
        # because it "needs" a specific ratio
        pad_image_to_aspect_ratio(image, image, 0.15)
        res = await get_inference(image)

    return DetectTableFormula.model_validate_json(
        res["message"]["content"]
    ).table_or_formula.lower()


async def extract_table(image: str) -> Table:
    """
    Extracts the contents of a Table, and returns it as headers and rows
    Args:
        image (str): the image url of table to be extracted
    Returns:
        Table: contents of table, as headers and rows
    """

    async def get_inference(image: str):
        res = await client.chat(
            model="minicpm-v-2",
            messages=[
                {
                    "role": "user",
                    "content": "give me the contents of this table",
                    "images": [convert_to_base64(image)],
                }
            ],
            format=Table.model_json_schema(),
            options={"temperature": 0},
            keep_alive=0,
        )
        return res

    try:
        res = await get_inference(image)
    except:
        # because it "needs" a specific ratio
        pad_image_to_aspect_ratio(image, image, 0.15)
        res = await get_inference(image)

    return Table.model_validate_json(res["message"]["content"])


async def extract_image(image: str) -> str:
    """
    Extracts the contents of a Image, and returns it as description
    Args:
        image (str): the image path to be extracted
    Returns:
        str: contents of Image
    """

    async def get_inference(image: str):
        res = await client.chat(
            model="minicpm-v-2",
            messages=[
                {
                    "role": "user",
                    "content": "give me the contents of this Image",
                    "images": [convert_to_base64(image)],
                }
            ],
            format=ImageDescription.model_json_schema(),
            keep_alive=0,
        )
        return res

    try:
        res = await get_inference(image)
    except:
        # because it "needs" a specific ratio
        pad_image_to_aspect_ratio(image, image, 0.15)
        res = await get_inference(image)

    return ImageDescription.model_validate_json(res["message"]["content"]).description


template_formula = """
I have a math formula extracted out,
it has an OCR based extraction,
this does not have valid math symbols
<valid-text>
{}
</valid-text>

this is a math based extraction latex format extraction
<valid-math>
{}
</valid-math>

give me a merged response, which has only the formula in the most accurate representation,
based on both information sources
"""


async def extract_formula(image: str) -> str:
    """
    Extracts formula as LaTeX from the given image
    Args:
        image (str): the image url of formula to be extracted
    Returns:
        str: string containing LaTeX formula
    """
    img = Image.open(image)
    output_ocr = pytesseract.image_to_string(img)
    output_1 = batch_inference([img], latex_model, latex_processor)

    res = await client.chat(
        model="marco-o1",
        messages=[
            {
                "role": "user",
                "content": template_formula.format(output_ocr, output_1),
            }
        ],
        format=Formula.model_json_schema(),
        options={"temperature": 0, "num_ctx": 4096},
        keep_alive=0,
    )

    text_content = LatexNodes2Text().latex_to_text(
        Formula.model_validate_json(res["message"]["content"]).formula
    )
    return text_content


system_text_instruct = """
You are a text assesser, I will provide you LaTeX based extracted text,
This text may contain invalid or mis-represented text symbols, 
these symbols could be mis-represented
using spelling errors or mis-spelt words

your role is to look
at the input text decide if it contains any such mis-representations
"""

template_latex = """
the following enclosed in tags is the LaTeX extraction,
<latex-text>
{}
</latex-text>
"""

template_ocr = """
the following is pure OCR extraction,
which may contain some invalid math symbols, but perfect text without any spelling issues,
<valid-text>
{}
</valid-text>

substitute the latex text mis-representations with valid text,
and return the final result
"""


async def extract_text(image: str) -> str:
    """
    Extracts text from image
    Args:
        image (str): the image url of text to be extracted
    Returns:
        str: string containing image text
    """
    latex_extract = batch_inference([Image.open(image)], latex_model, latex_processor)[
        0
    ]
    messages = [
        {
            "role": "system",
            "content": system_text_instruct,
        },
        {
            "role": "user",
            "content": template_latex.format(latex_extract),
        },
    ]

    ctx_size = (8192 * 2) if len(latex_extract) > 800 else 8192

    res = await client.chat(
        model="llama3.2",
        messages=messages,
        format=DocNeedsOCR.model_json_schema(),
        options={"temperature": 0, "num_ctx": ctx_size},
    )
    invalid_latex = DocNeedsOCR.model_validate_json(
        res["message"]["content"]
    ).invalid_latex

    print(latex_extract.strip())
    print(invalid_latex)

    if invalid_latex:
        tesseract_text = pytesseract.image_to_string(image)
        ctx_size = (8192 * 2) if len(tesseract_text) > 800 else 8192
        print(tesseract_text)
        messages.append(res.message)
        messages.append(
            {"role": "user", "content": template_ocr.format(tesseract_text)}
        )
        res = await client.chat(
            model="llama3.2",
            messages=messages,
            format=DocContent.model_json_schema(),
            options={"temperature": 0, "num_ctx": ctx_size},
            keep_alive=0,
        )
        result = DocContent.model_validate_json(res["message"]["content"]).text
        return result.strip()
    else:
        return LatexNodes2Text().latex_to_text(latex_extract)


# 1) Define a class to hold detection info
class DetectionInfo:
    def __init__(
        self,
        class_id: int,
        file_location: str = "",
    ):
        """
        class_id     = ID of the detected object
        confidence   = Confidence score
        xywh         = (centerX, centerY, width, height)
        file_location = This can be set later if needed
        """
        self.class_id = class_id
        self.file_location = file_location

    def set_file_location(self, location: str):
        self.file_location = location

    @property
    def top(self) -> float:
        """Return the 'top' (y-min) of the bounding box for sorting."""
        # xywh = (cx, cy, w, h)
        cx, cy, w, h = self.xywh
        return cy - h / 2.0

    @property
    def bottom(self) -> float:
        """Return the 'bottom' (y-max) of the bounding box."""
        # xywh = (cx, cy, w, h)
        cx, cy, w, h = self.xywh
        return cy + h / 2.0

    @property
    def left(self) -> float:
        """Return the 'left' (x-min) of the bounding box for sorting."""
        cx, cy, w, h = self.xywh
        return cx - w / 2.0

    @property
    def right(self) -> float:
        """Return the 'right' (x-max) of the bounding box."""
        cx, cy, w, h = self.xywh
        return cx + w / 2.0


async def build_content(idx, det, chapter_num, page_num, subject, content="") -> str:

    if det.class_id == 3:  # image data
        description = await extract_image(det.file_location)
        save_path = (
            f"outputs/{subject}/images/chapter_{chapter_num}/page_{page_num}_{idx}.jpg"
        )
        shutil.copyfile(src=det.file_location, dst=save_path)
        content += (
            f"""
        <image>
        <path>
        {save_path}
        </path>
        <description>
        {description}
        </description>
        </image>
        """
            + "\n"
        )
    elif det.class_id == 4:
        # caption text
        extracted_text = await extract_text(det.file_location)
        content += (
            f"""
            <caption>
            {extracted_text}
            </caption>
            """
            + "\n"
        )
    elif det.class_id == 8 or det.class_id == 5:
        table_or_formula = await classify_table_formula(det.file_location)
        is_table = "table" in table_or_formula

        if is_table:
            table = await extract_table(det.file_location)
            content += (
                f"""
                <table>
                {table.construct_table()}
                </table>
                """
                + "\n"
            )
        else:
            # model confuses between table and formula
            formula = await extract_formula(det.file_location)
            content += (
                f"""
            <formula>
            {formula}
            </formula>
            """
                + "\n"
            )
    elif det.class_id == 0:
        # title text
        extracted_text = await extract_text(det.file_location)
        content += (
            f"""
            <title>
            {extracted_text}
            </title>
            """
            + "\n"
        )
    else:
        # plain text
        extracted_text = await extract_text(det.file_location)
        content += (
            f"""
            <text>
            {extracted_text}
            </text>
            """
            + "\n"
        )
    return content

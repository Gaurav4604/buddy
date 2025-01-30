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

logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)

latex_model = load_model()
latex_processor = load_processor()


model = YOLOv10("inf.pt")

ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")


client = ollama.AsyncClient(host=ollama_url)


class PolishedDoc(BaseModel):
    markdown: str


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
        options={"temperature": 0.2},
        keep_alive=0,
    )

    text_content = LatexNodes2Text().latex_to_text(
        Formula.model_validate_json(res["message"]["content"]).formula
    )
    return text_content


system_merge = """
You are a pdf to text conversion merger,
your role is to look at a text extraction which has badly extracted math symbols
along with a valid math extraction for the same file
and merge the two files together, to build a common text, containing valid text
"""

template_files = """
I have a document page extracted out,
it has an OCR based extraction,
this does not have valid math symbols
<valid-text>
{}
</valid-text>

this is a math based extraction latex format extraction
<valid-math>
{}
</valid-math>

give me a merged response, with valid math, and retain all text info
"""


async def extract_text(image: str) -> str:
    """
    Extracts text from image
    Args:
        image (str): the image url of text to be extracted
    Returns:
        str: string containing image text
    """
    result = batch_inference([Image.open(image)], latex_model, latex_processor)
    text_content_latex = LatexNodes2Text().latex_to_text(result[0]).strip()

    tesseract_text = pytesseract.image_to_string(image)
    res = await client.chat(
        model="marco-o1",
        messages=[
            {
                "role": "system",
                "content": system_merge,
            },
            {
                "role": "user",
                "content": template_files.format(text_content_latex, tesseract_text),
            },
        ],
        format=DocContent.model_json_schema(),
        options={"temperature": 0.2},
        keep_alive=0,
    )
    result = DocContent.model_validate_json(res["message"]["content"]).text

    return LatexNodes2Text().latex_to_text(result)


# 1) Define a class to hold detection info
class DetectionInfo:
    def __init__(
        self,
        class_id: int,
        confidence: float,
        xywh: tuple[float, float, float, float],
        file_location: str = "",
    ):
        """
        class_id     = ID of the detected object
        confidence   = Confidence score
        xywh         = (centerX, centerY, width, height)
        file_location = This can be set later if needed
        """
        self.class_id = class_id
        self.confidence = confidence
        self.xywh = xywh
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

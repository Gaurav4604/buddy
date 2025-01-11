import cv2
from nougat import NougatModel
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.device import move_to_device
from pylatexenc.latex2text import LatexNodes2Text
import ollama
from pydantic import BaseModel
import pytesseract
import os
from PIL import Image
from doclayout_yolo import YOLOv10

# from texify.inference import batch_inference
# from texify.model.model import load_model
# from texify.model.processor import load_processor

latex_model = NougatModel.from_pretrained(get_checkpoint(None, model_tag="0.1.0-base"))
latex_model = move_to_device(latex_model, bf16=True, cuda=True)

# latex_model_sub = load_model()
# latex_processor_sub = load_processor()

from pix2tex.cli import LatexOCR


latex_model_sub = LatexOCR()

model = YOLOv10("inf.pt")


def save_detections(result: list):
    # Annotate and save the result
    annotated_frame = result[0].plot(pil=True, line_width=5, font_size=20)
    cv2.imwrite("result.jpg", annotated_frame)


class Table(BaseModel):
    headers: list[str]
    rows: list[list[str]]

    def construct_table(self) -> str:
        """constructs a table string, from the data on the object"""
        return f"""
    <headers>
    {str(self.headers)}
    </headers>
    <rows>
    {('<row>' + str(row) + '</row>' for row in self.rows)}
    </rows>
"""


def extract_table(image: str) -> Table:
    """
    Extracts the contents of a Table, and returns it as headers and rows
    Args:
        image (str): the image url of table to be extracted
    Returns:
        Table: contents of table, as headers and rows
    """
    res = ollama.chat(
        model="minicpm-v",
        messages=[
            {
                "role": "user",
                "content": "give me the contents of this table",
                "images": [image],
            }
        ],
        format=Table.model_json_schema(),
        options={"temperature": 0},
    )
    return Table.model_validate_json(res["message"]["content"])


def extract_formula(image: str) -> str:
    """
    Extracts formula as LaTeX from the given image
    Args:
        image (str): the image url of formula to be extracted
    Returns:
        str: string containing LaTeX formula
    """
    img = Image.open(image)

    output = latex_model.inference(image=img, early_stopping=False)
    output_sub = None
    if "Error" in output["predictions"][0] or output["predictions"][0].strip() == "":
        output_sub = latex_model_sub(img)
        output = None
        # output = batch_inference(
        #     [Image.open(img)], latex_model_sub, latex_processor_sub
        # )
    if output_sub is None:
        result = output["predictions"][0]
    else:
        result = output_sub
    text_content = LatexNodes2Text().latex_to_text(result)
    return text_content


def extract_text(image: str) -> str:
    """
    Extracts text from image
    Args:
        image (str): the image url of text to be extracted
    Returns:
        str: string containing image text
    """
    result = batch_inference([image], latex_model, latex_processor)[0]
    text_content_latex = LatexNodes2Text().latex_to_text(result).strip()

    tesseract_text = pytesseract.image_to_string(image)


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


def main():
    # Load the pre-trained model

    # Perform prediction
    img_path = "sample-eqn.png"
    det_res = model.predict(
        img_path,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cuda:0",  # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    print(det_res)

    # 2) Parse YOLO outputs and build a list of DetectionInfo objects
    all_detections: list[DetectionInfo] = []  # will hold DetectionInfo instances
    # Each 'det_res' element can contain multiple bounding boxes
    # so iterate through them
    for data in det_res:
        # data.boxes.cls -> tensor([...])
        # data.boxes.conf -> tensor([...])
        # data.boxes.xywh -> tensor([...]) with shape (N,4)

        classes = data.boxes.cls.tolist()  # convert to python list
        confidences = data.boxes.conf.tolist()
        xywhs = data.boxes.xywh.tolist()  # list of [cx, cy, w, h]

        for cls_id, conf, box_xywh in zip(classes, confidences, xywhs):
            if cls_id != 2:
                det_info = DetectionInfo(
                    class_id=int(cls_id),
                    confidence=float(conf),
                    xywh=tuple(box_xywh),
                    file_location="",  # can set later if you wish
                )
                all_detections.append(det_info)

    # 3) Sort all detections from top to bottom, then left to right
    #    We sort primarily by `top`, secondarily by `left`.
    all_detections.sort(key=lambda det: (det.top, det.left))

    # 4) Extract each snippet and 5) save into /temp folder
    # Create the /temp folder if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    # Read the original image using PIL
    original_img = Image.open(img_path).convert("RGB")

    for idx, det in enumerate(all_detections):

        # Crop the image
        snippet = original_img.crop((det.left, det.top, det.right, det.bottom))

        # Save snippet
        snippet_path = f"temp/snippet_{idx}.png"
        snippet.save(snippet_path)
        det.set_file_location(snippet_path)

        print(f"Saved snippet {idx} -> {snippet_path}")

    for det in all_detections:
        if det.class_id == 8:
            print(det.file_location)
            formula = extract_formula(det.file_location)
            # print("formula: " + formula)
            print(formula)
        if det.class_id == 5:
            table = extract_table(det.file_location)
            print("table: " + str(table.model_dump_json()))
        else:
            print()


if __name__ == "__main__":
    main()

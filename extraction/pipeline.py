import cv2
from utils import (
    DetectionInfo,
    model,
    extract_formula,
    extract_table,
    extract_text,
    extract_image,
)
import os
import shutil
from PIL import Image
from pdf2image import convert_from_path


def extraction_pipeline(input_img: str, page_num: int):
    # Load the pre-trained model

    det_res = model.predict(
        input_img,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cuda:0",  # Device to use (e.g., 'cuda:0' or 'cpu')
    )

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

    # for outputs from the pages
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(os.path.join("outputs", "images")):
        os.makedirs(os.path.join("outputs", "images"))
    if not os.path.exists(os.path.join("outputs", "pages")):
        os.makedirs(os.path.join("outputs", "pages"))

    # Read the original image using PIL
    original_img = Image.open(input_img).convert("RGB")

    for idx, det in enumerate(all_detections):

        # Crop the image
        snippet = original_img.crop((det.left, det.top, det.right, det.bottom))

        # Save snippet
        snippet_path = f"temp/snippet_{idx}.png"
        snippet.save(snippet_path)
        det.set_file_location(snippet_path)

        print(f"Saved snippet {idx} -> {snippet_path}")

    content = ""

    for idx, det in enumerate(all_detections):
        if det.class_id == 3:  # image data
            description = extract_image(det.file_location)
            save_path = f"outputs/images/page_{page_num}_{idx}.jpg"
            shutil.copyfile(dst=save_path, src=det.file_location)
            content += """
            <image>
            <path>
            {}
            </path>
            <description>
            {}
            </description>
            </image>
            """.format(
                save_path, description
            )
        elif det.class_id == 8:
            formula = extract_formula(det.file_location)
            content += formula + "\n"
        elif det.class_id == 5:
            table = extract_table(det.file_location)
            content += table.construct_table() + "\n"
        else:
            text = extract_text(det.file_location)
            content += text + "\n"

    with open(f"outputs/pages/page_{page_num}.txt", "w", encoding="utf-8") as f:
        f.write(content)

    shutil.rmtree("temp")
    return content


def extraction_pipeline_from_pdf(pdf_path: str) -> str:
    """
    Converts each page of the given PDF into an image and calls the
    existing extraction_pipeline() on each page image. Aggregates all
    extracted content into a single string.
    """
    # Convert PDF to a list of PIL Images
    pages = convert_from_path(pdf_path, dpi=300)

    os.makedirs("pages", exist_ok=True)
    all_pages_content = []
    for i, page in enumerate(pages):
        # Save the current PDF page as a PNG
        page_filename = f"pages/page_{i}.png"
        page.save(page_filename, "PNG")

        # Now run the existing pipeline function
        page_content = extraction_pipeline(page_filename, page_num=i)
        all_pages_content.append(f"--- Page {i+1} ---\n{page_content}")

    shutil.rmtree("temp")
    # Combine text from all pages
    return "\n".join(all_pages_content)


if __name__ == "__main__":
    print(extraction_pipeline_from_pdf("files/automata_2.pdf"))

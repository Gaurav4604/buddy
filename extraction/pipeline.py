from utils import (
    DetectionInfo,
    model,
    classify_table_formula,
    extract_formula,
    extract_table,
    extract_text,
    extract_image,
    save_detections,
    pad_image_to_aspect_ratio,
)
import os
import shutil
from PIL import Image
from pdf2image import convert_from_path
from typing import Any
import asyncio
import json


async def get_content(idx, det, chapter_num, page_num, content="") -> str:

    if det.class_id == 3:  # image data
        description = await extract_image(det.file_location)
        save_path = f"outputs/images/chapter_{chapter_num}/page_{page_num}_{idx}.jpg"
        print(f"image {save_path} - {idx} - {description}")
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
        print(f"caption {extracted_text} - {idx}")
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
        print(table_or_formula)
        is_table = "table" in table_or_formula

        if is_table:
            table = await extract_table(det.file_location)
            print(f"table {table} - {idx}")
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
            print(f"formula {formula} - {idx}")
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
        print(f"title {extracted_text} - {idx}")
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
        print(f"text {extracted_text} - {idx}")
        content += (
            f"""
            <text>
            {extracted_text}
            </text>
            """
            + "\n"
        )
    return content


async def extraction_pipeline(input_img: str, chapter_num: int, page_num: int):
    # 1) Load the pre-trained model predictions
    det_res = model.predict(
        input_img,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cuda:0",
    )

    # 2) Read the original image using PIL
    original_img = Image.open(input_img).convert("RGB")
    save_detections(det_res, input_img)

    # 3) Parse YOLO outputs and build a list of DetectionInfo objects
    all_detections: list[DetectionInfo] = []

    # Each 'det_res' element can contain multiple bounding boxes
    # so iterate through them
    for data in det_res:
        classes = data.boxes.cls.tolist()  # e.g. tensor([ ... ]) -> Python list
        confidences = data.boxes.conf.tolist()
        xywhs = data.boxes.xywh.tolist()  # list of [cx, cy, w, h]

        for cls_id, conf, box_xywh in zip(classes, confidences, xywhs):
            # Skip if class is "abandon" (id == 2)
            if cls_id == 2:
                continue

            # Build DetectionInfo
            det_info = DetectionInfo(
                class_id=int(cls_id),
                confidence=float(conf),
                xywh=tuple(box_xywh),
                file_location="",
            )
            all_detections.append(det_info)

    # 4) Sort all detections top-to-bottom, then left-to-right
    all_detections.sort(key=lambda det: (det.top, det.left))

    # 5) Extract each snippet and save into /temp folder
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(os.path.join("outputs", "images")):
        os.makedirs(os.path.join("outputs", "images"))
    if not os.path.exists(os.path.join("outputs", "images", f"chapter_{chapter_num}")):
        os.makedirs(os.path.join("outputs", "images", f"chapter_{chapter_num}"))

    if not os.path.exists(os.path.join("outputs", "pages")):
        os.makedirs(os.path.join("outputs", "pages"))
    if not os.path.exists(os.path.join("outputs", "pages", f"chapter_{chapter_num}")):
        os.makedirs(os.path.join("outputs", "pages", f"chapter_{chapter_num}"))

    for idx, det in enumerate(all_detections):
        # Crop the snippet
        snippet = original_img.crop((det.left, det.top, det.right, det.bottom))
        snippet_path = f"temp/snippet_{idx}.png"
        snippet.save(snippet_path)
        det.set_file_location(snippet_path)
        print(f"Saved snippet {idx} -> {snippet_path}")

    # 6) Build textual content for this page
    content = ""
    for idx, det in enumerate(all_detections):
        # {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
        # content +=
        max_retries = 3
        execution_timeout = 30 if not (det.class_id == 8 or det.class_id == 5) else 45
        for attempt in range(1, max_retries + 1):
            # Recreate the task each time we retry
            task = asyncio.create_task(
                get_content(idx, det, chapter_num, page_num, content)
            )

            try:
                content = await asyncio.wait_for(task, timeout=execution_timeout)
                print(content)
                # If we get here, it succeeded within 45s — break out of the loop
                break

            except asyncio.TimeoutError:
                # Cancel the timed-out task
                task.cancel()

                if attempt < max_retries:
                    execution_timeout += 15
                    print(f"Attempt {attempt} timed out. Retrying...")
                else:
                    print(
                        f"Attempt {attempt} timed out. Maximum retries reached; giving up."
                    )
                    # No more retries; you can handle it (e.g. return, raise, etc.)
                    # break or raise an exception, depending on your needs
                    exit()

    # 7) Write the page's content to disk
    with open(
        f"outputs/pages/chapter_{chapter_num}/page_{page_num + 1}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)

    # Clean up /temp
    shutil.rmtree("temp")
    return content


async def async_extraction_pipeline_from_pdf(
    pdf_path: str, chapter_num: int = 0, start_page: int = 0
) -> str:
    """
    Async version of 'extraction_pipeline_from_pdf' that converts pages to images,
    then awaits 'extraction_pipeline(...)' on each page (which is also async).
    Returns combined text from all pages.
    """
    # Convert only the pages from start_page onward
    pages = convert_from_path(pdf_path, dpi=300, first_page=(start_page + 1))

    os.makedirs("pages", exist_ok=True)
    all_pages_content = []

    for i, page in enumerate(pages):
        real_page_index = start_page + i
        page_filename = f"pages/page_{real_page_index}.png"
        page.save(page_filename, "PNG")

        # Instead of calling asyncio.run(...), we just 'await' the pipeline call
        page_content = await extraction_pipeline(
            page_filename, chapter_num, page_num=real_page_index
        )

        # Accumulate your results if needed
        all_pages_content.append(f"--- Page {real_page_index + 1} ---\n{page_content}")

    page_count_save(chapter_num, len(pages))
    return "\n".join(all_pages_content)


def page_count_save(chapter_num: int = 0, page_count: int = 0):
    # File name
    file_name = "outputs/pages/chapters_tree.json"

    # Check if the file exists; if not, create an empty dictionary
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            data = json.load(file)
    else:
        data = {}

    # Add the new chapter_num as a key with the list of page numbers
    if chapter_num >= 0 and page_count > 0:
        data[str(chapter_num)] = list(range(1, page_count + 1))

    # Save the updated data back to the file
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Saved chapter {chapter_num} with page count {page_count} to {file_name}.")


async def main():
    cpt_1 = await async_extraction_pipeline_from_pdf(
        "files/automata_cpt_1.pdf", chapter_num=0
    )
    cpt_2 = await async_extraction_pipeline_from_pdf(
        "files/automata_cpt_2.pdf", chapter_num=1
    )
    return [cpt_1, cpt_2]


if __name__ == "__main__":
    asyncio.run(main())

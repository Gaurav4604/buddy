from utils import (
    DetectionInfo,
    model,
    extract_formula,
    extract_table,
    extract_text,
    extract_image,
    save_detections,
)
import os
import shutil
from PIL import Image
from pdf2image import convert_from_path
from typing import Any
import asyncio


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
    elif det.class_id == 8:
        # Possibly a formula
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
    elif det.class_id == 5:
        # Possibly a table
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

    # -- Helper functions for bounding boxes --

    def bbox_area(left: float, top: float, right: float, bottom: float) -> float:
        """Returns area of the bounding box defined by [left, top, right, bottom]."""
        return max(0.0, right - left) * max(0.0, bottom - top)

    def compute_overlap_ratio(
        new_box: tuple[float, float, float, float],
        existing_box: tuple[float, float, float, float],
    ) -> float:
        """
        Returns the fraction of 'new_box' area overlapped by 'existing_box'.
        Overlap ratio = overlap_area / area(new_box).
        """
        n_left, n_top, n_right, n_bottom = new_box
        e_left, e_top, e_right, e_bottom = existing_box

        # Intersection coords
        inter_left = max(n_left, e_left)
        inter_top = max(n_top, e_top)
        inter_right = min(n_right, e_right)
        inter_bottom = min(n_bottom, e_bottom)

        if inter_right < inter_left or inter_bottom < inter_top:
            # No overlap
            return 0.0

        overlap_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        new_area = bbox_area(n_left, n_top, n_right, n_bottom)

        if new_area == 0:
            return 0.0

        return overlap_area / new_area

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

            # If it's class_id == 3, we add it straight away (following your original logic).
            # Else we check bounding box overlaps.
            if cls_id == 3:
                all_detections.append(det_info)
            else:
                # Convert YOLO xywh -> [left, top, right, bottom]
                cx, cy, w, h = box_xywh
                left = cx - w / 2
                top = cy - h / 2
                right = cx + w / 2
                bottom = cy + h / 2
                new_box = (left, top, right, bottom)

                # Compare with existing detections
                skip_this_box = False
                for existing_det in all_detections:
                    # Convert existing_det xywh -> [left, top, right, bottom]
                    e_cx, e_cy, e_w, e_h = existing_det.xywh
                    e_left = e_cx - e_w / 2
                    e_top = e_cy - e_h / 2
                    e_right = e_cx + e_w / 2
                    e_bottom = e_cy + e_h / 2
                    existing_box = (e_left, e_top, e_right, e_bottom)

                    # If the new box is 99% or more overlapped by an existing box, skip it
                    overlap_ratio = compute_overlap_ratio(new_box, existing_box)
                    if overlap_ratio >= 0.99:
                        skip_this_box = True
                        break

                if not skip_this_box:
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
        for attempt in range(1, max_retries + 1):
            # Recreate the task each time we retry
            task = asyncio.create_task(
                get_content(idx, det, chapter_num, page_num, content)
            )

            try:
                content = await asyncio.wait_for(task, timeout=45)
                print(content)
                # If we get here, it succeeded within 45s — break out of the loop
                break

            except asyncio.TimeoutError:
                # Cancel the timed-out task
                task.cancel()

                if attempt < max_retries:
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

    return "\n".join(all_pages_content)


if __name__ == "__main__":
    # Now do a single call to asyncio.run with the entire multi-page process.
    output = asyncio.run(
        async_extraction_pipeline_from_pdf("files/automata_cpt_2.pdf", chapter_num=1)
    )
    print(output)

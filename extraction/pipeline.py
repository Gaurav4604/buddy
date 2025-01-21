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


def extraction_pipeline(input_img: str, page_num: int):
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
    if not os.path.exists(os.path.join("outputs", "pages")):
        os.makedirs(os.path.join("outputs", "pages"))

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
        if det.class_id == 3:  # image data
            description = extract_image(det.file_location)
            save_path = f"outputs/images/page_{page_num}_{idx}.jpg"
            print(f"image {save_path} - {idx} - {description}")
            shutil.copyfile(src=det.file_location, dst=save_path)
            content += f"""
            <image>
            <path>
            {save_path}
            </path>
            <description>
            {description}
            </description>
            </image>
            """
        elif det.class_id == 8:
            # Possibly a formula
            formula = extract_formula(det.file_location)
            print(f"formula {formula} - {idx}")
            content += formula + "\n"
        elif det.class_id == 5:
            # Possibly a table
            table = extract_table(det.file_location)
            print(f"table {table} - {idx}")
            content += table.construct_table() + "\n"
        else:
            # General text
            extracted_text = extract_text(det.file_location)
            print(f"text {extracted_text} - {idx}")
            content += extracted_text + "\n"

    # 7) Write the page's content to disk
    with open(f"outputs/pages/page_{page_num}.txt", "w", encoding="utf-8") as f:
        f.write(content)

    # Clean up /temp
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

        # Call the pipeline for this image
        page_content = extraction_pipeline(page_filename, page_num=i)
        all_pages_content.append(f"--- Page {i+1} ---\n{page_content}")

    # Combine text from all pages
    return "\n".join(all_pages_content)


if __name__ == "__main__":
    print(extraction_pipeline_from_pdf("files/automata.pdf"))

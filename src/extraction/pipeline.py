from .structure_extraction.top_down import top_down_pipeline
from .structure_extraction.research import research_structure_pipeline

from .utils import DetectionInfo, build_content
import os
import shutil
from pdf2image import convert_from_path
import asyncio
import json
import time


async def extraction_pipeline(
    input_img: str,
    chapter_num: int,
    page_num: int,
    document_structure: str,
    subject: str,
    manual_terminate: str,
):
    if document_structure == "research":
        [original_img, final_boxes] = research_structure_pipeline(input_img=input_img)
    else:
        [original_img, final_boxes] = top_down_pipeline(input_img=input_img)

    # 6) Create the output directories if not exist
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    all_detections: list[DetectionInfo] = []
    # 7) Crop out each merged box region
    #    Because you originally wanted to do snippet extraction
    #    from YOLO bounding boxes, now we do it from merged boxes:
    for idx, (tl, br, cls_id) in enumerate(final_boxes):
        x1, y1 = tl
        x2, y2 = br
        snippet = original_img.crop((x1, y1, x2, y2))
        snippet_path = f"temp/snippet_{idx}.png"
        snippet.save(snippet_path)

        all_detections.append(
            DetectionInfo(class_id=cls_id, file_location=snippet_path)
        )

    if not os.path.exists(os.path.join("outputs", f"{subject}")):
        os.makedirs(os.path.join("outputs", f"{subject}"))
    if not os.path.exists(os.path.join("outputs", f"{subject}", "images")):
        os.makedirs(os.path.join("outputs", f"{subject}", "images"))
    if not os.path.exists(
        os.path.join("outputs", f"{subject}", "images", f"chapter_{chapter_num}")
    ):
        os.makedirs(
            os.path.join("outputs", f"{subject}", "images", f"chapter_{chapter_num}")
        )

    if not os.path.exists(os.path.join("outputs", f"{subject}", "pages")):
        os.makedirs(os.path.join("outputs", f"{subject}", "pages"))
    if not os.path.exists(
        os.path.join("outputs", f"{subject}", "pages", f"chapter_{chapter_num}")
    ):
        os.makedirs(
            os.path.join("outputs", f"{subject}", "pages", f"chapter_{chapter_num}")
        )
    # # 6) Build textual content for this page
    content = ""
    terminate_loop = False
    for idx, det in enumerate(all_detections):
        if not terminate_loop:
            # {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
            max_retries = 3
            execution_timeout = (
                60 if not (det.class_id == 8 or det.class_id == 5) else 150
            )
            for attempt in range(1, max_retries + 1):
                # Recreate the task each time we retry
                task = asyncio.create_task(
                    build_content(
                        idx,
                        det,
                        chapter_num,
                        page_num,
                        subject,
                        content,
                        manual_terminate,
                    )
                )

                try:
                    if attempt > 1 and (det.class_id == 8 or det.class_id == 5):
                        det.class_id = 3  # force detection to fall back as image, for easier detection
                        task = asyncio.create_task(
                            build_content(
                                idx,
                                det,
                                chapter_num,
                                page_num,
                                subject,
                                content,
                                manual_terminate,
                            )
                        )
                    [content, manually_terminated] = await asyncio.wait_for(
                        task, timeout=execution_timeout
                    )
                    terminate_loop = manually_terminated
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
                        break

    # 7) Write the page's content to disk
    with open(
        f"outputs/{subject}/pages/chapter_{chapter_num}/page_{page_num + 1}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)

    # # Clean up /temp
    shutil.rmtree("temp")
    return [content, terminate_loop]


async def async_extraction_pipeline_from_pdf(
    pdf_path: str,
    subject: str = "general",
    chapter_num: int = 0,
    document_structure: str = "top_down",
    start_page: int = 0,
    manual_terminate: str = "",
) -> str:
    """
    Async version of 'extraction_pipeline_from_pdf' that converts pages to images,
    then awaits 'extraction_pipeline(...)' on each page (which is also async).
    Returns combined text from all pages.
    """
    # Convert only the pages from start_page onward
    pages = convert_from_path(pdf_path, dpi=300, first_page=(start_page + 1))

    pages_consumed = 0

    os.makedirs("pages", exist_ok=True)
    all_pages_content = []

    print("Starting Extraction...")

    for i, page in enumerate(pages):
        start_time = time.time()

        real_page_index = start_page + i
        page_filename = f"pages/page_{real_page_index}.png"
        page.save(page_filename, "PNG")

        [page_content, manually_terminated] = await extraction_pipeline(
            page_filename,
            chapter_num,
            subject=subject,
            page_num=real_page_index,
            document_structure=document_structure,
            manual_terminate=manual_terminate,
        )
        # Accumulate your results if needed
        all_pages_content.append(f"--- Page {real_page_index + 1} ---\n{page_content}")

        end_time = time.time()
        print(f"time taken for extraction --- {end_time - start_time} s---")
        pages_consumed += 1
        if manually_terminated:
            break

    if len(manual_terminate) > 0:
        page_count_save(subject, chapter_num, pages_consumed)
    else:
        page_count_save(subject, chapter_num, len(pages))

    shutil.rmtree("pages")
    return "\n".join(all_pages_content)


def page_count_save(subject: str, chapter_num: int = 0, page_count: int = 0):
    # File name
    file_name = f"outputs/{subject}/pages/chapters_tree.json"

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

import os
import cv2
import json
import numpy as np
from pdf2image import convert_from_path


def convert_pdf_to_images(pdf_path):
    """
    Convert PDF pages to images using pdf2image.
    """
    pages = convert_from_path(pdf_path, dpi=500)  # Convert PDF to a list of PIL images
    return pages


def extract_and_save_bbox(image, bbox, filename, save_dir):
    """
    Extract the region within the bounding box and save the image.
    """
    x1, y1, x2, y2 = map(int, bbox)  # Convert bounding box coordinates to integers
    cropped_image = image[y1:y2, x1:x2]  # Extract the region of the image
    image_path = os.path.join(save_dir, filename)
    cv2.imwrite(image_path, cropped_image)  # Save the image
    return image_path


def process_detections(image, detections, page_dir):
    """
    Process detections to find 'Picture' and its associated caption.
    Extracts and saves both the picture and caption as images.
    Returns a list of dictionaries containing 'picture' and 'caption' file paths.
    """
    results_list = []
    picture_index = 0  # Keep track of the picture count

    if detections.boxes is None:
        print("No detections found.")
        return results_list

    # Get the bounding boxes and class names
    xyxy_boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding boxes as numpy array
    class_ids = detections.boxes.cls.cpu().numpy()  # Class IDs as numpy array
    class_names = [detections.names[int(cls_id)] for cls_id in class_ids]  # Class names

    # Combine the boxes and class names into a single list of tuples
    detection_data = list(zip(xyxy_boxes, class_names))

    # Sort detection data based on the top y-coordinate (bbox[1]) to process from top to bottom
    detection_data.sort(key=lambda x: x[0][1])

    # Loop through sorted detections and find 'Picture'
    for bbox, class_name in detection_data:
        picture_filename = f"{picture_index}.png"
        picture_path = extract_and_save_bbox(image, bbox, picture_filename, page_dir)
        x1, y1, x2, y2 = map(int, bbox)
        results_list.append(
            {
                "filename": picture_filename,
                "path": picture_path,
                "class": class_name,
                "bbox": [x1, y1, x2, y2],
            }
        )
        picture_index += 1

    return results_list


def process_pdf_with_yolo(pdf_path, model):
    """
    Process each page of the PDF, detect 'Picture' and its caption, and save results.
    Creates a subdirectory for each page to store the images.
    Returns a list of dictionaries with image paths for each page.
    """
    pages = convert_pdf_to_images(pdf_path)  # Convert PDF pages to images
    all_results = []

    for page_number, page_image in enumerate(pages):
        print(f"Processing Page {page_number}...")

        # Create a subdirectory for each page
        page_dir = os.path.join("images", f"page_{page_number}")
        os.makedirs(page_dir, exist_ok=True)

        # Convert PIL image to OpenCV format
        image_cv = np.array(page_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Run the YOLO model on the image to detect objects
        results = model(source=image_cv, conf=0.5, iou=0.8)[0]

        page_results = process_detections(image_cv, results, page_dir)
        print(page_results)
        all_results.append({"page": page_number, "results": page_results})

        # Save the results for the page as a JSON file
        json_path = os.path.join(page_dir, f"page_{page_number}_results.json")
        with open(json_path, "w") as json_file:
            json.dump(
                {"page": page_number, "results": page_results}, json_file, indent=4
            )

    return all_results


# # Example usage
from ultralytics import YOLO

# # Load the YOLOv10 model
model = YOLO("document_semantic_inference.pt").to("cuda")

# # Path to your PDF file
pdf_path = "files/automata.pdf"

# Process the PDF with YOLO and print the results
final_results = process_pdf_with_yolo(pdf_path, model)
print(final_results)

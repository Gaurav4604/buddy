import os
from PIL import Image
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from pylatexenc.latex2text import LatexNodes2Text

# Define the path containing images
path = "images/page_5"

# Get all image file paths in the directory
image_files = [
    os.path.join(path, file)
    for file in os.listdir(path)
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
]

# Load all images as PIL.Image objects
imgs = [Image.open(image_file) for image_file in image_files]

# Load the model and processor
model = load_model()
processor = load_processor()

# Perform batch inference
results = batch_inference(imgs, model, processor)

# Convert LaTeX to text for each result and print
for index, result in enumerate(results):
    text_content = LatexNodes2Text().latex_to_text(result)
    print(f"Text content for image {image_files[index]}:\n{text_content}\n")

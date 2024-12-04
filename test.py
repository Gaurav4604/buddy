from PIL import Image

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor


img = Image.open("formula.jpg")
model = load_model()
processor = load_processor()
results = batch_inference([img], model, processor)
print(results)


import os
from pylatexenc.latex2text import LatexNodes2Text

text_content = LatexNodes2Text().latex_to_text(results[0])
print(text_content)

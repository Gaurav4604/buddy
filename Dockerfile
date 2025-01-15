FROM python:3.9-slim-bullseye

WORKDIR /app

# # Install necessary dependencies including OpenGL libraries for OpenCV
# RUN apt-get update && apt-get install -y \
#     python3-dev \
#     python3-pip \
#     build-essential \
#     libffi-dev \
#     libssl-dev \
#     git \
#     bash \
#     docker.io \
#     rustc \
#     cargo \
#     tesseract-ocr \
#     poppler-utils \
#     libgl1 \
#     && apt-get clean

# # Create a virtual environment named 'buddy' for Python dependencies
# RUN python3 -m venv /opt/buddy

# # Activate the virtual environment and upgrade pip
# RUN /opt/buddy/bin/pip install --upgrade pip

# # Set the virtual environment to be the default for any subsequent commands
# ENV PATH="/opt/buddy/bin:$PATH"

# # Install Python dependencies
# COPY requirements.txt /app/requirements.txt
# COPY document_semantic_inference.pt /app/document_semantic_inference.pt

# RUN pip install --no-cache-dir -r /app/requirements.txt

# # Install model data from Pix2Text
# RUN python -c "from pix2text import Pix2Text; Pix2Text()"



# # Create directories for cache and output
# RUN mkdir -p /app/cache /app/rough /app/convert /app/tex /app/txt

# # Copy the Python script to the container
# COPY ./convert/* /app/convert/

# # Command to run the script
# CMD ["bash", "-c", "python /app/convert/pdf_to_txt.py && python /app/convert/txt_to_tex.py && python /app/convert/tex_to_txt.py"]

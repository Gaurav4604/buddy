FROM python:3.9-slim-bullseye

WORKDIR /app

# # Install necessary dependencies including OpenGL libraries for OpenCV
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    bash \
    docker.io \
    rustc \
    cargo \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && apt-get clean


RUN pip install --upgrade pip


# # Install Python dependencies
COPY requirements.txt /app/requirements.txt
COPY inf.pt /app/inf.pt

# # Create directories for cache and output
# RUN mkdir -p /app/cache /app/rough /app/convert /app/tex /app/txt

# # Copy the Python script to the container
COPY ./extraction/* /app/extraction/

# # Command to run the script
CMD ["bash", "-c", "python /app/extraction/pipeline.py"]

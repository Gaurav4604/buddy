import ollama
from PIL import Image
import requests
from io import BytesIO

# Load the image from a URL or local file
image_path = "./image.png"  # Replace with your image path or URL

# If you're using an image URL
# image_url = "https://example.com/image.png"
# response = requests.get(image_url)
# image = Image.open(BytesIO(response.content))

# Using a local image file
image = Image.open(image_path)

# Use the ollama library to query the 'llava-phi3' model
# The image file is passed directly to the model along with a prompt
response = ollama.chat(
    model="llava-phi3",
    messages=[
        {
            "role": "user",
            "content": "Describe the content of this image. considering that this is a state machine",
            "images": [image_path],
        }
    ],
)

# Print the model's response
print(response["message"]["content"])

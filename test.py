import asyncio
import ollama
import base64
import requests

client = ollama.Client(host="http://127.0.0.1:11434")


def convert_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode("utf-8")
        return base64_string


image = "E:\\Programming\\LLMs\\buddy\\temp\\snippet_21.png"


# async def main():
#     # res = await client.chat(
#     #     model="minicpm-v-2",
#     #     messages=[
#     #         {
#     #             "role": "user",
#     #             "content": "is this a table? or a formula?",
#     #             "images": [
#     #                 convert_to_base64(
#     #                     image
#     #                 )
#     #             ],
#     #         }
#     #     ],
#     #     options={"temperature": 0},
#     #     keep_alive=0,
#     # )


# asyncio.run(main())
# res = client.generate(
#     model="hf.co/bartowski/Qwen2-VL-7B-Instruct-GGUF:Q4_0",
#     prompt="give me the contents of this Image, in english",
#     images=[convert_to_base64(image)],
#     options={
#         "temperature": 0,
#         # "num_predict": 200,
#         "frequency_penalty": 1.0,
#     },
#     keep_alive=0,
# )

# print(res)
# bs64img = convert_to_base64(image)


# res = requests.post(
#     "http://localhost:11434/api/generate",
#     json={
#         "model": "llava-llama3",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "give me the contents of this Image",
#                 "images": [bs64img],
#             }
#         ],
#         "stream": False,
#     },
# )

# print(res.content)
from PIL import Image, ImageOps


def pad_image_to_aspect_ratio(image_path, output_path, target_aspect_ratio):
    # Open the original image
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Calculate target dimensions
    target_width = original_width
    target_height = int(target_width * target_aspect_ratio)

    if target_height < original_height:
        target_height = original_height
        target_width = int(target_height / target_aspect_ratio)

    # Calculate padding
    padding_left = (target_width - original_width) // 2
    padding_top = (target_height - original_height) // 2
    padding_right = target_width - original_width - padding_left
    padding_bottom = target_height - original_height - padding_top

    # Apply padding
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    padded_img = ImageOps.expand(img, padding, fill="white")

    # Save the padded image
    padded_img.save(output_path)


# Usage
pad_image_to_aspect_ratio(
    "E:\\Programming\\LLMs\\buddy\\temp\\snippet_20.png",
    "E:\\Programming\\LLMs\\buddy\\temp\\snippet_20_padded.png",
    0.15,
)

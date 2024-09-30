import ollama
import os
import re


rough_folder = "rough"

output_dir = "cleaned"
count = 1
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(rough_folder):
    file_path = os.path.join(rough_folder, file_name)

    # Skip if it's not a text file
    if not file_path.endswith(".txt"):
        continue

    # Read the content of the text file
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    # Define the prompt for LLaMA 3.1
    prompt = (
        f'Here is the text from my file: "{file_content}". this is a physics based text document.'
        "fix all symbols based on file data "
        "DO NOT PROVIDE ANYTHING EXCEPT FIXED FILE DATA, NO EXTRA INFO OR DESCRIPTION NEEDED"
    )

    res = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    output = res["message"]["content"]
    text_file_name = os.path.join(output_dir, f"{count}.txt")
    count += 1
    with open(text_file_name, "w", encoding="utf-8") as text_file:
        text_file.write(output)
    print(output)

from ollama import Client
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
base_url = OLLAMA_URL
ollama = Client(host=base_url)

output_dir = "./tex"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


input_dir = "./raw/"
count = 0

system_prompt = f"convert the data to tex file format only, display nothing else, other than the required tex"

for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    if not file_path.endswith(".txt"):
        continue

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    prompt = f"""this is a tex based extraction latex format extraction: `{file_content}`
         convert it to proper latex document, ensure that it has tags such as begin, section, subsection and has usepackage amsmath"""

    res = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "options": {system_prompt: system_prompt},
            },
            {
                "role": "user",
                "content": "remove everything except the latex document, no extra info required, not even ``` to specific markdown syntax",
                "options": {system_prompt: system_prompt},
            },
        ],
    )

    output = res["message"]["content"]
    print(file_name)
    fn = file_name.split(".txt")[0]
    print(fn)
    text_file_name = os.path.join(output_dir, f"{fn}.tex")
    count += 1
    with open(text_file_name, "w", encoding="utf-8") as text_file:
        text_file.write(output)
    print(output)

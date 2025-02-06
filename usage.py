script = open("src/main.py", "r", encoding="utf-8")

prompt = f"""
give me a usage markdown for the following file,
considering it is responsible for consuming docs, 
generating questions on docs, evaluating answers on doc, 
and asking questions to do

ensure to include code blocks, with flags and examples,
also keep it lightweight and fun with emojis

<file>
{script.read()}
</file>"""

import ollama


res = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
    options={"num_ctx": 16384, "temperature": 0},
)

print(res.message.content, file=open("usage.md", "w", encoding="utf-8"))

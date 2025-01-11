from markitdown import MarkItDown
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434", api_key="bruh")

md = MarkItDown(
    llm_client=client,
    llm_model="minicpm-v",
)
result = md.convert("images/page_2/6.png")
print(result.text_content)

import pymupdf4llm

md_text = pymupdf4llm.to_markdown("file.pdf")

import pathlib

pathlib.Path("output.md").write_bytes(md_text.encode())

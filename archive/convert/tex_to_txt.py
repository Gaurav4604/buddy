import os
from pylatexenc.latex2text import LatexNodes2Text

# Directory containing the .tex files
tex_dir = "tex"
output_dir = "txt"  # Directory to save the converted .txt files

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all .tex files in the directory
for filename in os.listdir(tex_dir):
    if filename.endswith(".tex"):
        tex_file_path = os.path.join(tex_dir, filename)

        # Read the contents of the .tex file
        with open(tex_file_path, "r", encoding="utf-8") as tex_file:
            tex_content = tex_file.read()

        # Convert LaTeX to plain text
        text_content = LatexNodes2Text().latex_to_text(tex_content)

        # Define output file path
        output_file_path = os.path.join(
            output_dir, f"{os.path.splitext(filename)[0]}.txt"
        )

        # Save the converted text to .txt file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(text_content)

        print(f"Converted {filename} to {output_file_path}")

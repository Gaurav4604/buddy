from utils import RAGtoolkit
from os.path import isfile
import json


def dump_extraction_to_db():
    file_name = "outputs/pages/chapters_tree.json"
    with open(file_name, "r") as file:
        chapter_tree = json.load(file)
        for chapter in chapter_tree.keys():
            overall_kit = RAGtoolkit(chapter_num=chapter)

            for page in chapter_tree[chapter]:

                chapter_kit = RAGtoolkit(int(chapter), page)
                print(chapter_kit.chapter_name, chapter_kit.page_number)

                print(
                    isfile(
                        f"outputs/pages/{chapter_kit.chapter_name}/{chapter_kit.page_number}.txt"
                    )
                )
                with open(
                    f"outputs/pages/{chapter_kit.chapter_name}/{chapter_kit.page_number}.txt",
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = f.read()
                    chunks = chapter_kit.generate_chunks(data)

                    chapter_kit.add_docs(chunks)

                    print(f"--- page {page} - chunks added ---")
            print(f"--- chapter {chapter} - chunks added ---")

            metadata = overall_kit.generate_meta(
                f"outputs/pages/{chapter_kit.chapter_name}"
            )
            overall_kit.add_meta_data(
                metadata["tags"], metadata["topics"], metadata["summary"]
            )


if __name__ == "__main__":
    dump_extraction_to_db()

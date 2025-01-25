from utils import RAGtoolkit
from os.path import isfile


def dump_extraction_to_db():
    chapter_tree = {0: [1, 2, 3, 4, 5, 6], 1: [1, 2, 3, 4, 5, 6, 7]}

    for chapter in chapter_tree.keys():
        overall_kit = RAGtoolkit(chapter_num=chapter)

        for page in chapter_tree[chapter]:

            chapter_kit = RAGtoolkit(chapter, page)
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
        overall_kit.add_meta_data(metadata["tags"], metadata["summary"])


page_1_kit = RAGtoolkit()

result = page_1_kit.query_docs("tell me about on/off switch")

print(result["metadatas"])

for r in result["documents"][0]:
    print(r)
    print("------------")


print(page_1_kit.get_summary())

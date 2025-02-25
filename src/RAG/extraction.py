from .utils import RAGtoolkit
from os.path import isfile
from os import remove
import json
from flask_socketio import SocketIO


def dump_extraction_to_db(topic: str = "general", socketio: SocketIO = None):
    file_name = f"outputs/{topic}/pages/chapters_tree.json"
    with open(file_name, "r") as file:
        chapter_tree = json.load(file)
        for chapter in chapter_tree.keys():
            for page in chapter_tree[chapter]:

                chapter_kit = RAGtoolkit(int(chapter), page, topic=topic)
                print(chapter_kit.chapter_name, chapter_kit.page_number)

                print(
                    isfile(
                        f"outputs/{topic}/pages/{chapter_kit.chapter_name}/page_{chapter_kit.page_number}.txt"
                    )
                )
                with open(
                    f"outputs/{topic}/pages/{chapter_kit.chapter_name}/page_{chapter_kit.page_number}.txt",
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = f.read()
                    if len(data) > 0:
                        chunks = chapter_kit.generate_chunks(data)

                        chapter_kit.add_docs(chunks)
                        if socketio:
                            socketio.emit(
                                "doc_extraction",
                                {
                                    "message": f"Page {chapter_kit.page_number} chunks Generated"
                                },
                            )
                        summary = chapter_kit.generate_summary(data)
                        chapter_kit.add_meta(
                            summary.summary,
                            "[" + ",".join(summary.tags) + "]",
                            chapter_kit.chapter_num,
                            chapter_kit.page_number,
                        )

                        if socketio:
                            socketio.emit(
                                "doc_extraction",
                                {
                                    "message": f"Page {chapter_kit.page_number} summary Generated"
                                },
                            )

                    print(f"--- page {page} - chunks added ---")
            print(f"--- chapter {chapter} - chunks added ---")
    remove(file_name)

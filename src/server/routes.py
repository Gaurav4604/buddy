import os
from flask import request, jsonify
from flask_cors import CORS
from server import app, socketio
from tools import read_pipeline
import asyncio
from server.database import GlobalsDB

# import time

# Define a directory to save uploads
UPLOAD_FOLDER = os.path.join(os.getcwd(), "files")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add CORS to allow frontend connections
CORS(app, resources={r"/*": {"origins": "*"}})


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@app.route("/topics", methods=["GET", "POST"])
def topics():
    db = GlobalsDB()
    if request.method == "GET":
        topics = db.fetch_all_unique_topics()
        return jsonify(topics), 200

    elif request.method == "POST":
        # Create a new topic.
        data = request.get_json()
        if not data or "topic_name" not in data:
            return jsonify({"status": "error", "message": "Missing topic_name"}), 400

        topic_name = data["topic_name"]
        success = db.add_topic(topic_name)
        if success:
            # Optionally, return the created topic details.
            return jsonify({"topic_name": topic_name}), 201
        else:
            return jsonify({"status": "error", "message": "Could not add topic"}), 500


@app.route("/topics/<topic_name>", methods=["GET"])
def topic_details(topic_name):
    db = GlobalsDB()
    if request.method == "GET":
        # Fetch chapter content for the provided topic.
        # If no content exists, an empty list is returned.
        content = db.fetch_topic_chapters(topic_name)
        return jsonify(content), 200

    elif request.method == "POST":
        # Insert new topic content for the given topic.
        data = request.get_json()
        required_fields = ["chapter_num", "summary", "tags"]
        if not data or not all(key in data for key in required_fields):
            return (
                jsonify({"status": "error", "message": "Missing required fields"}),
                400,
            )

        chapter_num = data["chapter_num"]
        summary = data["summary"]
        tags = data["tags"]
        success = db.add_topic_content(topic_name, chapter_num, summary, tags)
        if success:
            return jsonify({"status": "success", "message": "Topic content added"}), 201
        else:
            return (
                jsonify({"status": "error", "message": "Could not add topic content"}),
                500,
            )


@app.route("/topics/<topic_name>", methods=["POST"])
def update_topic_content(topic_name):
    """
    Endpoint to update topic content for a specific topic.
    Expects one or more PDF files along with a 'chapter_num' in the form data.
    It saves the PDF, emits a SocketIO event, processes the PDF to generate tags and summary,
    and then updates the database with the results.
    """
    # Validate PDF files upload
    if "pdfs" not in request.files:
        return jsonify({"status": "error", "message": "No files uploaded."}), 400

    pdf_files = request.files.getlist("pdfs")
    metadata = request.form.to_dict()  # Should contain chapter_num
    if "chapter_num" not in metadata:
        return (
            jsonify(
                {"status": "error", "message": "Missing chapter_num in form data."}
            ),
            400,
        )

    try:
        chapter_num = int(metadata["chapter_num"])
    except ValueError:
        return (
            jsonify({"status": "error", "message": "Invalid chapter_num value."}),
            400,
        )

    # Emit a SocketIO event to signal the start of processing
    socketio.emit(
        "doc_extraction",
        {"message": f"Extraction for file {pdf_files[0].filename} Started"},
        namespace="/",
    )
    print(f"Emitted doc_extraction event for {pdf_files[0].filename}")

    saved_files = []
    for pdf in pdf_files:
        if pdf.filename == "":
            continue  # Skip files with no filename
        if not pdf.filename.lower().endswith(".pdf"):
            continue  # Optionally skip non-PDF files

        file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
        pdf.save(file_path)
        saved_files.append(pdf.filename)

    if not saved_files:
        return jsonify({"status": "error", "message": "No valid PDF files found."}), 400

    # Run the heavy asynchronous pipeline to extract tags and summary
    tags, summary = asyncio.run(
        read_pipeline(
            topic=topic_name,  # Topic from the URL
            file_path=f"files/{saved_files[0]}",
            chapter_num=chapter_num,  # Chapter number from the form data
            socketio=socketio,
            doc_structure="default",
            manual_terminate="",
        )
    )

    # Update the database with the generated tags and summary
    db = GlobalsDB()
    success = db.add_topic_content(topic_name, chapter_num, summary, tags)
    if not success:
        return (
            jsonify(
                {"status": "error", "message": "Failed to update topic content in DB."}
            ),
            500,
        )

    return jsonify({"status": "success", "tags": tags, "summary": summary}), 200


# If you run this file directly, start the server
if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

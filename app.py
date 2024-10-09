from fastapi import FastAPI, Request
import requests
import os
import ollama
from ollama import Client

# Initialize FastAPI app
app = FastAPI()

# Get the Ollama service URL from environment variable or use default Docker Compose URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

model_name = "llama3.2"
base_url = OLLAMA_URL

client = Client(host=base_url)


@app.post("/process_input/")
async def process_input(request: Request):
    """
    This endpoint accepts POST requests with a prompt as JSON input
    and sends the prompt to the Ollama service for processing.
    """
    # Extract JSON body from request
    request_data = await request.json()
    prompt = request_data.get("prompt", "")

    if not prompt:
        return {"error": "No prompt provided"}

    # Send the prompt to Ollama
    try:
        response = client.chat(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        response_data = response["message"]["content"]

        # Return the response from Ollama
        return {"ollama_response": response_data}

    except Exception as e:
        return {"error": str(e)}


# Root endpoint for testing
@app.get("/")
def read_root():
    return {
        "message": "API is running. Use the /process_input/ endpoint to send input."
    }

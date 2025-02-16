from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
app.config["SECRET_KEY"] = "pointless-secret-key"

# Enable CORS for all HTTP routes
CORS(app)

# Create the SocketIO server with CORS allowed for SocketIO events
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Import routes so that they register with the app
from server import routes

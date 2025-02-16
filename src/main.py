from server import app, socketio

if __name__ == "__main__":
    # Run the Flask app via SocketIO.
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

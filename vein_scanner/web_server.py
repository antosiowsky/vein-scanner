# web_server.py

import logging
import threading
from flask import Flask, Response, render_template_string, request # <-- ADD 'request'
import cv2
import requests

class WebServerThread(threading.Thread):
    def __init__(self, host='0.0.0.0', port=5000):
        super().__init__()
        self.daemon = True
        self.host = host
        self.port = port
        self.flask_app = Flask(__name__)
        self._frame = None
        self._frame_lock = threading.Lock()
        self.add_routes()

    def add_routes(self):
        @self.flask_app.route('/')
        def index():
            return render_template_string("""
                <html><body style="background-color:#111;text-align:center;">
                <img src="{{ url_for('video_feed') }}" width="90%"></body></html>
            """)

        @self.flask_app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # --- THIS IS THE CRITICAL FIX ---
        @self.flask_app.route('/shutdown', methods=['POST'])
        def shutdown():
            """This endpoint triggers the server's internal shutdown mechanism."""
            shutdown_func = request.environ.get('werkzeug.server.shutdown')
            if shutdown_func is None:
                logging.error('Not running with the Werkzeug Server')
                return "Server shutdown failed: Not a Werkzeug server."
            
            logging.info("Shutdown endpoint called. Stopping Flask server.")
            shutdown_func()
            return "Server is shutting down..."

    def generate_frames(self):
        while True:
            with self._frame_lock:
                if self._frame is None: continue
                (flag, encodedImage) = cv2.imencode(".jpg", self._frame)
                if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')

    def update_frame(self, frame):
        with self._frame_lock:
            self._frame = frame.copy()

    def run(self):
        logging.info(f"Starting web server on http://{self.host}:{self.port}")
        self.flask_app.run(host=self.host, port=self.port, debug=False, threaded=True)
        logging.info("Web server has stopped.")

    def stop(self):
        """Stops the Flask web server by sending a request to the shutdown endpoint."""
        try:
            requests.post(f'http://127.0.0.1:{self.port}/shutdown')
        except requests.exceptions.ConnectionError:
            logging.info("Web server was not running or already stopped.")
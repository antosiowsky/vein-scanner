# vein_scanner_app.py

import time
import logging
import os
import threading

from camera_processor import ParamSetter, CameraThread
from menu_system import MenuSystem
from web_server import WebServerThread

class VeinScannerApp:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self._shutdown_event = threading.Event()
        
        self.param_setter = ParamSetter()
        self.param_setter.load_params_from_file()
        
        self.camera_thread = CameraThread(self.param_setter)
        self.web_server_thread = None

        self.menu = MenuSystem(
            param_setter=self.param_setter,
            start_scan_callback=self.camera_thread.start_processing,
            stop_scan_callback=self.camera_thread.stop_processing,
            start_web_server_callback=self.start_web_server,
            stop_web_server_callback=self.stop_web_server,
            shutdown_callback=self.shutdown
        )
        logging.info("Vein Scanner Application Initialized.")

    # --- NEW METHODS to manage the web server ---
    def start_web_server(self):
        if self.web_server_thread and self.web_server_thread.is_alive():
            logging.warning("Web server is already running.")
            return
        logging.info("Starting web server thread...")
        self.web_server_thread = WebServerThread()
        self.web_server_thread.start()
        self.camera_thread.set_web_server(self.web_server_thread) # Link it to the camera

    def stop_web_server(self):
        if not self.web_server_thread or not self.web_server_thread.is_alive():
            logging.warning("Web server is not running.")
            return
        logging.info("Stopping web server thread...")
        self.web_server_thread.stop()
        self.web_server_thread.join(timeout=3.0)
        self.camera_thread.set_web_server(None) # Unlink from camera
        self.web_server_thread = None
        
    def run(self):
        logging.info("Starting background camera thread...")
        self.camera_thread.start()
        try:
            self._shutdown_event.wait()
        except KeyboardInterrupt:
            logging.info("Program interrupted by user (Ctrl+C).")
        finally:
            self.cleanup()

    def shutdown(self):
        logging.info("Shutdown requested from menu. Signaling main loop to exit...")
        self._shutdown_event.set()
    
    def cleanup(self):
        logging.info("Starting application cleanup...")
        
        # --- NEW: Stop web server during cleanup ---
        self.stop_web_server()

        if self.camera_thread and self.camera_thread.is_alive():
            logging.info("Stopping camera thread...")
            self.camera_thread.stop()
            self.camera_thread.join(timeout=3.0)
        
        if self.menu: self.menu.cleanup()
        
        if self._shutdown_event.is_set():
            logging.info("Cleanup complete. Executing system shutdown.")
            # os.system("sudo shutdown -h now")

if __name__ == "__main__":
    app = VeinScannerApp()
    app.run()
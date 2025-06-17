# vein_scanner_app.py

import time
import logging
import os

from camera_processor import ParamSetter, CameraThread
from menu_system import MenuSystem

class VeinScannerApp:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # --- KEY CHANGE: Initialize Camera and Parameters first ---
        logging.info("Initializing camera and parameters...")
        self.param_setter = ParamSetter()
        self.param_setter.load_params_from_file()

        # The CameraThread is now a long-lived background process.
        # It's created once and runs for the lifetime of the app.
        self.camera_thread = CameraThread(self.param_setter)
        
        # --- KEY CHANGE: Pass camera's control methods directly to the menu ---
        # The menu will now talk directly to the camera thread to start/stop
        # the *activity* of scanning, not the thread itself.
        self.menu = MenuSystem(
            param_setter=self.param_setter,
            start_scan_callback=self.camera_thread.start_processing,
            stop_scan_callback=self.camera_thread.stop_processing, 
            shutdown_callback=self.shutdown
        )
        
        logging.info("Vein Scanner Application Initialized.")

    def run(self):
        """Main loop of the application."""
        logging.info("Starting background camera thread...")
        self.camera_thread.start() # Start the background thread
        
        try:
            # The main thread just needs to stay alive to handle shutdown.
            # All work is done in the camera and menu/GPIO threads.
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Program interrupted by user (Ctrl+C).")
        finally:
            self.cleanup()

    def shutdown(self):
        """Callback for shutting down the Pi."""
        logging.info("Shutdown requested. Initiating cleanup...")
        self.cleanup()
        logging.info("Shutting down the system.")
        # os.system("sudo shutdown -h now") # Uncomment for production
        exit()

    def cleanup(self):
        logging.info("Starting application cleanup...")
        # Clean up the menu system first (e.g., GPIO pins)
        if self.menu:
            self.menu.cleanup()
            
        # Stop and join the camera thread
        if self.camera_thread and self.camera_thread.is_alive():
            logging.info("Stopping camera thread...")
            self.camera_thread.stop()
            self.camera_thread.join(timeout=2.0)
            if self.camera_thread.is_alive():
                logging.warning("Camera thread did not stop in time.")
        
        logging.info("Application cleanup complete. Exiting.")

# Removed start_scan and stop_scan methods as they are no longer needed here.
# Their logic has moved into the CameraThread.

if __name__ == "__main__":
    app = VeinScannerApp()
    app.run()
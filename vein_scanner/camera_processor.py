# camera_processor.py

import cv2
import json
import os
import logging
import threading
import time
import numpy as np

# ImageProcessing and ParamSetter classes are unchanged and correct.
class ImageProcessing:
    @staticmethod
    def apply_clahe(frame_img, for_value, clip_limit, tile_grid_size, median_ksize=3):
        lab_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
        for _ in range(int(for_value)):
            l_channel = clahe.apply(l_channel)
        lab_img = cv2.merge((l_channel, a_channel, b_channel))
        median_lab_img = cv2.medianBlur(lab_img, median_ksize)
        return cv2.cvtColor(median_lab_img, cv2.COLOR_LAB2BGR)

class ParamSetter:
    CONFIG_FILE_PATH = "config.json"; DEFAULT_CONFIG_FILE_PATH = "default_config.json"
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.DEFAULT_PARAMS = self.load_default_params_from_file()
        self.clahe_for_value = self.DEFAULT_PARAMS["for_value"]; self.clahe_clip_limit = self.DEFAULT_PARAMS["cliplimit_value"]
        self.clahe_tile_grid_size = self.DEFAULT_PARAMS["tilegrid_value"]; self.gain = self.DEFAULT_PARAMS["gain"]
        self.exposure = self.DEFAULT_PARAMS["exposure"]; self.invert_colors = self.DEFAULT_PARAMS["invert_colors"]
    def load_default_params_from_file(self):
        if os.path.exists(self.DEFAULT_CONFIG_FILE_PATH):
            with open(self.DEFAULT_CONFIG_FILE_PATH, "r") as f: return json.load(f)
        else:
            params = {"for_value": 2, "cliplimit_value": 2.5, "tilegrid_value": 8, "gain": 10, "exposure": 100, "invert_colors": False}
            with open(self.DEFAULT_CONFIG_FILE_PATH, "w") as f: json.dump(params, f)
            return params
    def save_params_to_file(self):
        params = {"for_value": self.clahe_for_value, "cliplimit_value": self.clahe_clip_limit, "tilegrid_value": self.clahe_tile_grid_size, "gain": self.gain, "exposure": self.exposure, "invert_colors": self.invert_colors}
        with open(self.CONFIG_FILE_PATH, "w") as f: json.dump(params, f)
    def load_params_from_file(self):
        params = self.DEFAULT_PARAMS.copy()
        if os.path.exists(self.CONFIG_FILE_PATH):
            with open(self.CONFIG_FILE_PATH, "r") as f: params.update(json.load(f))
        self.clahe_for_value = params["for_value"]; self.clahe_clip_limit = params["cliplimit_value"]; self.clahe_tile_grid_size = params["tilegrid_value"]
        self.gain = params["gain"]; self.exposure = params["exposure"]; self.invert_colors = params.get("invert_colors", False)

class CameraThread(threading.Thread):
    def __init__(self, param_setter):
        super().__init__()
        self.param_setter = param_setter
        self.name = "CameraThread"
        self._processing_active = threading.Event()
        self._stop_event = threading.Event()
        self.daemon = True
        self.cap = None
        self.window_name = "Vein Scanner Feed"
        self.web_server = None

    def set_web_server(self, server_instance):
        self.web_server = server_instance

    def run(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise IOError("Cannot open camera.")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'GREY'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            logging.info("Camera initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            return

        # Create a normal, resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720) 

        # --- BLACK FRAME LOGIC RESTORED ---
        # Create a black image for the local display window
        black_frame_local = np.zeros((720, 960, 3), dtype=np.uint8)
        # Create a black image for the web feed
        black_frame_web = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Initially, show the black screen locally # <-- RESTORED
        cv2.imshow(self.window_name, black_frame_local)
        cv2.waitKey(1)

        while not self._stop_event.is_set():
            logging.info("Camera thread is idle. Waiting for scan command...")
            self._processing_active.wait()
            if self._stop_event.is_set():
                break

            logging.info("Scan started. Applying hardware settings...")
            self.param_setter.load_params_from_file()
            self._apply_camera_settings()
            
            while self._processing_active.is_set() and not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                if len(frame.shape) == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                processed_frame = ImageProcessing.apply_clahe(
                    frame, 
                    self.param_setter.clahe_for_value, 
                    self.param_setter.clahe_clip_limit, 
                    self.param_setter.clahe_tile_grid_size
                )
                
                if self.param_setter.invert_colors:
                    processed_frame = cv2.bitwise_not(processed_frame)
                
                cv2.imshow(self.window_name, processed_frame)
                
                if self.web_server and self.web_server.is_alive():
                    web_frame = cv2.resize(processed_frame, (1280, 960))
                    self.web_server.update_frame(web_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_processing()
            
            # --- BEHAVIOR ON STOP RESTORED ---
            if not self._stop_event.is_set():
                # Show the black frame on the local display
                cv2.imshow(self.window_name, black_frame_local)
                logging.info("Scan stopped. Displaying black screen locally.")

                # Also show black screen on the web feed
                if self.web_server and self.web_server.is_alive():
                    self.web_server.update_frame(black_frame_web)
                
                cv2.waitKey(1) # Ensure the window updates

        # Final Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera thread shut down and resources released.")

    def _apply_camera_settings(self):
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_GAIN, float(self.param_setter.gain))
            self.cap.set(cv2.CAP_PROP_EXPOSURE, int(self.param_setter.exposure))
            logging.info(f"Set GAIN={self.param_setter.gain}, EXPOSURE={self.param_setter.exposure}")
        except Exception as e:
            logging.warning(f"Could not apply camera settings: {e}")
    
    def start_processing(self):
        if self.is_alive():
            self._processing_active.set()

    def stop_processing(self):
        self._processing_active.clear()

    def stop(self):
        self._stop_event.set()
        self._processing_active.set()
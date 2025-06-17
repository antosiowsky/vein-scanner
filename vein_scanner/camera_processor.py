# camera_processor.py

import cv2
import json
import os
import logging
import threading
import time
import numpy as np

# ImageProcessing class is unchanged and works as you provided.
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


# --- UPDATED ParamSetter Class ---
class ParamSetter:
    CONFIG_FILE_PATH = "config.json"
    DEFAULT_CONFIG_FILE_PATH = "default_config.json"
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.DEFAULT_PARAMS = self.load_default_params_from_file()
        self.clahe_for_value = self.DEFAULT_PARAMS["for_value"]
        self.clahe_clip_limit = self.DEFAULT_PARAMS["cliplimit_value"]
        self.clahe_tile_grid_size = self.DEFAULT_PARAMS["tilegrid_value"]
        self.gain = self.DEFAULT_PARAMS["gain"]
        self.exposure = self.DEFAULT_PARAMS["exposure"]
        self.invert_colors = self.DEFAULT_PARAMS["invert_colors"] # <-- ADDED

    def load_default_params_from_file(self):
        if os.path.exists(self.DEFAULT_CONFIG_FILE_PATH):
            with open(self.DEFAULT_CONFIG_FILE_PATH, "r") as config_file: return json.load(config_file)
        else:
            default_params = {
                "for_value": 2, 
                "cliplimit_value": 2.5, 
                "tilegrid_value": 8, 
                "gain": 10, 
                "exposure": 100,
                "invert_colors": False
            }
            with open(self.DEFAULT_CONFIG_FILE_PATH, "w") as config_file: json.dump(default_params, config_file)
            return default_params

    def save_params_to_file(self):
        params = {
            "for_value": self.clahe_for_value, 
            "cliplimit_value": self.clahe_clip_limit, 
            "tilegrid_value": self.clahe_tile_grid_size, 
            "gain": self.gain, 
            "exposure": self.exposure,
            "invert_colors": self.invert_colors
        }
        with open(self.CONFIG_FILE_PATH, "w") as config_file: json.dump(params, config_file)
        logging.info(f"Parameters saved to {self.CONFIG_FILE_PATH}")

    def load_params_from_file(self):
        params = self.DEFAULT_PARAMS.copy()
        if os.path.exists(self.CONFIG_FILE_PATH):
            with open(self.CONFIG_FILE_PATH, "r") as config_file:
                loaded_params = json.load(config_file)
                params.update(loaded_params)
        self.clahe_for_value = params["for_value"]
        self.clahe_clip_limit = params["cliplimit_value"]
        self.clahe_tile_grid_size = params["tilegrid_value"]
        self.gain = params["gain"]
        self.exposure = params["exposure"]
        self.invert_colors = params.get("invert_colors", False) # <-- ADDED (.get for safety)


# --- UPDATED CameraThread Class ---
class CameraThread(threading.Thread):
    def __init__(self, param_setter):
        super().__init__()
        # ... (rest of __init__ is unchanged)
        self.param_setter = param_setter
        self.name = "CameraThread"
        self._processing_active = threading.Event()
        self._stop_event = threading.Event()
        self.daemon = True 
        self.cap = None
        self.window_name = "Vein Scanner Feed"

    def run(self):
        # ... (initialization and window creation are unchanged)
        logging.info("Camera thread starting up...")
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened(): raise IOError("Cannot open camera.")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'GREY'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            logging.info("Camera initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}"); return

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imshow(self.window_name, black_frame)
        cv2.waitKey(1)

        while not self._stop_event.is_set():
            self._processing_active.wait()
            if self._stop_event.is_set(): break
            
            self.param_setter.load_params_from_file()
            self._apply_camera_settings()
            
            while self._processing_active.is_set() and not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret: logging.warning("Failed to read frame..."); time.sleep(0.1); continue

                if len(frame.shape) == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                processed_frame = ImageProcessing.apply_clahe(frame, self.param_setter.clahe_for_value, self.param_setter.clahe_clip_limit, self.param_setter.clahe_tile_grid_size)

                # --- NEW INVERT LOGIC ---
                if self.param_setter.invert_colors:
                    processed_frame = cv2.bitwise_not(processed_frame)

                resized_frame = cv2.resize(processed_frame, (1920, 1080))
                cv2.imshow(self.window_name, resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): self.stop_processing()

            if not self._stop_event.is_set():
                cv2.imshow(self.window_name, black_frame)
                cv2.waitKey(1)

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera thread shut down completely.")

    def _apply_camera_settings(self):
        # ... (this method is unchanged)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            gain_val = float(self.param_setter.gain)
            self.cap.set(cv2.CAP_PROP_GAIN, gain_val)
            exposure_val = int(self.param_setter.exposure)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
            logging.info(f"Set GAIN={gain_val}, EXPOSURE={exposure_val}")
        except Exception as e:
            logging.warning(f"Could not apply camera settings: {e}")
    
    def start_processing(self):
        if self.is_alive(): self._processing_active.set()
    def stop_processing(self):
        self._processing_active.clear()
    def stop(self):
        self._stop_event.set(); self._processing_active.set()
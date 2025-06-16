import cv2
import json
import os
import logging

# ------------------ ImageProcessing CLASS ------------------
class ImageProcessing:
    @staticmethod
    def apply_clahe(frame_img, for_value, clip_limit, tile_grid_size, median_ksize=3):
        """Apply CLAHE and median blur to the given frame."""
        lab_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)

        for _ in range(int(for_value)):
            clahe = cv2.createCLAHE(
                clipLimit=float(clip_limit),
                tileGridSize=(int(tile_grid_size), int(tile_grid_size))
            )
            l_channel = clahe.apply(l_channel)

        lab_img = cv2.merge((l_channel, a_channel, b_channel))
        median_lab_img = cv2.medianBlur(lab_img, median_ksize)

        return cv2.cvtColor(median_lab_img, cv2.COLOR_LAB2BGR)

# ------------------ ParamSetter CLASS ------------------
class ParamSetter:
    CONFIG_FILE_PATH = "config.json"
    DEFAULT_CONFIG_FILE_PATH = "default_config.json"

    def __init__(self):
        self.DEFAULT_PARAMS = self.load_default_params_from_file()
        self.clahe_for_value = self.DEFAULT_PARAMS["for_value"]
        self.clahe_clip_limit = self.DEFAULT_PARAMS["cliplimit_value"]
        self.clahe_tile_grid_size = self.DEFAULT_PARAMS["tilegrid_value"]
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    def load_default_params_from_file(self):
        """Load default parameters from a JSON file."""
        if os.path.exists(self.DEFAULT_CONFIG_FILE_PATH):
            with open(self.DEFAULT_CONFIG_FILE_PATH, "r") as config_file:
                default_params = json.load(config_file)
                logging.info(f"Default parameters loaded from {self.DEFAULT_CONFIG_FILE_PATH}")
                return default_params
        else:
            default_params = {
                "for_value": 2,
                "cliplimit_value": 2.5,
                "tilegrid_value": 8
            }
            with open(self.DEFAULT_CONFIG_FILE_PATH, "w") as config_file:
                json.dump(default_params, config_file)
            logging.info(f"Default parameters created in {self.DEFAULT_CONFIG_FILE_PATH}")
            return default_params
    def save_params_to_file(self):
        """Save current parameters to a JSON file."""
        params = {
            "for_value": self.clahe_for_value,
            "cliplimit_value": self.clahe_clip_limit,
            "tilegrid_value": self.clahe_tile_grid_size
        }
        with open(self.CONFIG_FILE_PATH, "w") as config_file:
            json.dump(params, config_file)
        logging.info(f"Parameters saved to {self.CONFIG_FILE_PATH}")

    def load_params_from_file(self):
        """Load parameters from a JSON file or fallback to default."""
        if os.path.exists(self.CONFIG_FILE_PATH):
            with open(self.CONFIG_FILE_PATH, "r") as config_file:
                params = json.load(config_file)
                self.clahe_for_value = params.get("for_value", self.DEFAULT_PARAMS["for_value"])
                self.clahe_clip_limit = params.get("cliplimit_value", self.DEFAULT_PARAMS["cliplimit_value"])
                self.clahe_tile_grid_size = params.get("tilegrid_value", self.DEFAULT_PARAMS["tilegrid_value"])
                logging.info(f"Parameters loaded from {self.CONFIG_FILE_PATH}")
        else:
            logging.warning("No config file found, loading default parameters.")
            self.clahe_for_value = self.DEFAULT_PARAMS["for_value"]
            self.clahe_clip_limit = self.DEFAULT_PARAMS["cliplimit_value"]
            self.clahe_tile_grid_size = self.DEFAULT_PARAMS["tilegrid_value"]

# ------------------ MAIN FUNCTION ------------------
def main():
    param_setter = ParamSetter()
    param_setter.load_params_from_file()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'GREY'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    if not cap.isOpened():
        logging.error("Cannot open camera.")
        return

    logging.info("Press 'q' to quit.")

    # Create fullscreen window
    window_name = "CLAHE Processed Image"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from camera.")
            break

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        processed_frame = ImageProcessing.apply_clahe(
            frame,
            param_setter.clahe_for_value,
            param_setter.clahe_clip_limit,
            param_setter.clahe_tile_grid_size
        )

        # Resize to 1920x1080 before display
        resized_frame = cv2.resize(processed_frame, (1920, 1080))
        cv2.imshow(window_name, resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import os
import cv2
import requests
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PlateDetector:
    def __init__(self, cascade_path=None):
        """
        Initializes the license plate detector.
        Downloads the cascade XML if it doesn't exist.
        """
        if cascade_path is None:
            self.cascade_path = config.HAAR_CASCADE_NAME
        else:
            self.cascade_path = cascade_path
            
        self._ensure_cascade_exists()
        
        # Load the cascade classifier
        self.plate_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.plate_cascade.empty():
            raise IOError(f"Failed to load Haar Cascade from path: {self.cascade_path}")
            
    def _ensure_cascade_exists(self):
        """
        Downloads the Haar Cascade XML file from OpenCV repository if not present.
        """
        if not os.path.exists(self.cascade_path):
            logging.info(f"Haar Cascade file not found. Downloading from {config.HAAR_CASCADE_URL}...")
            try:
                response = requests.get(config.HAAR_CASCADE_URL, timeout=15)
                response.raise_for_status()
                with open(self.cascade_path, "wb") as f:
                    f.write(response.content)
                logging.info("Cascade file downloaded successfully.")
            except Exception as e:
                logging.error(f"Error downloading cascade file: {e}")
                # Fallback to check if it's in a package subdirectory or raise exception
                raise RuntimeError(
                    f"Could not retrieve Haar Cascade XML. Please download it manually from "
                    f"{config.HAAR_CASCADE_URL} and save it as '{self.cascade_path}' in the working directory."
                ) from e

    def detect_plates(self, image, scale_factor=1.1, min_neighbors=4, min_size=(50, 15)):
        """
        Detects license plates in the input image.
        
        Parameters:
        - image: OpenCV image (BGR)
        - scale_factor: Parameter specifying how much the image size is reduced at each image scale.
        - min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        - min_size: Minimum possible object size. Objects smaller than that are ignored.
        
        Returns:
        - list of dicts: Each dict contains 'box' (x, y, w, h) and 'crop' (numpy array of cropped plate region).
        """
        if image is None:
            logging.error("Input image is None")
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect plates
        plates = self.plate_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size
        )
        
        results = []
        for (x, y, w, h) in plates:
            # Crop the detected plate region from the original image
            crop = image[y:y+h, x:x+w]
            results.append({
                "box": (x, y, w, h),
                "crop": crop
            })
            
        logging.info(f"Detected {len(results)} potential license plates.")
        return results

    @staticmethod
    def draw_annotations(image, detections):
        """
        Draws bounding boxes and metadata annotations on the image.
        
        Parameters:
        - image: The original BGR image.
        - detections: List of detection results, where each contains 'box' and optionally 'text'.
        
        Returns:
        - annotated_image: Copy of the original image with annotations drawn.
        """
        annotated_image = image.copy()
        for det in detections:
            x, y, w, h = det["box"]
            # Draw green bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw text label if it exists
            label = det.get("text", "Plate")
            confidence = det.get("confidence", None)
            if confidence is not None:
                label = f"{label} ({confidence:.0f}%)"
                
            # Place text background
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_image, 
                (x, y - text_h - 10), 
                (x + text_w, y), 
                (0, 255, 0), 
                cv2.FILLED
            )
            
            # Write text
            cv2.putText(
                annotated_image, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2, 
                cv2.LINE_AA
            )
            
        return annotated_image

if __name__ == "__main__":
    # Quick self-test check
    try:
        detector = PlateDetector()
        print("PlateDetector initialized successfully and Haar Cascade loaded.")
    except Exception as ex:
        print(f"Error during initialization: {ex}")

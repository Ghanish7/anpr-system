import cv2
import numpy as np
import pytesseract
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set the Tesseract executable path from config
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

class OCREngine:
    def __init__(self):
        """
        Initializes the OCR Engine. Verifies if Tesseract is accessible.
        """
        try:
            # Quick test of tesseract version to check if path is correct
            version = pytesseract.get_tesseract_version()
            logging.info(f"Tesseract OCR initialized successfully. Version: {version}")
        except Exception as e:
            logging.warning(
                f"Tesseract OCR is not found at '{config.TESSERACT_CMD}'. "
                f"Please ensure Tesseract is installed and the path in config.py is correct.\nError: {e}"
            )

    def preprocess_plate(self, crop):
        """
        Preprocesses the license plate cropped image to optimize for OCR:
        - Grayscale conversion
        - Resizing to standard high resolution
        - Bilateral filter for noise reduction while keeping edges sharp
        - Otsu's binarization for clear thresholding
        - Background analysis to ensure black text on white background (inversion if needed)
        - Morphological cleaning
        """
        if crop is None or crop.size == 0:
            return None

        # 1. Grayscale conversion
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. Resize to increase character resolution (Tesseract prefers letters to be at least 30-40px high)
        # We scale up by a factor of 2.5
        resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        # 3. Apply Bilateral Filter (reduces noise while preserving edges)
        filtered = cv2.bilateralFilter(resized, 11, 17, 17)

        # 4. Apply Otsu's Thresholding to get a binary image
        thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 5. Check if background is dark and text is light, and invert if necessary.
        # Tesseract performs significantly better when text is black on white.
        # We estimate background color by checking average intensity of pixels along the border.
        h, w = thresh.shape
        border_pixels = []
        border_pixels.extend(thresh[0, :])          # Top row
        border_pixels.extend(thresh[h-1, :])        # Bottom row
        border_pixels.extend(thresh[1:h-1, 0])      # Left col
        border_pixels.extend(thresh[1:h-1, w-1])    # Right col

        mean_border = np.mean(border_pixels)
        if mean_border < 127:
            # Dark background, light text -> Invert to get black text on white background
            thresh = cv2.bitwise_not(thresh)

        # 6. Morphological opening to clean small noise dots inside/outside letters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return processed

    def recognize_text(self, crop):
        """
        Runs Tesseract OCR on the plate crop.
        Tries both PSM 7 (single line of text) and PSM 8 (single word) for best matching.
        Uses character whitelist to limit character set to uppercase letters and digits.
        
        Returns:
        - text (str): The recognized string
        - confidence (float): Average confidence score (0 to 100)
        """
        processed_crop = self.preprocess_plate(crop)
        if processed_crop is None:
            return "", 0.0

        # Whitelist uppercase A-Z and digits 0-9
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        # Config options: PSM 7 (treat image as single line of text), whitelist characters
        custom_config_psm7 = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        custom_config_psm8 = f"--psm 8 -c tessedit_char_whitelist={whitelist}"

        text, confidence = "", 0.0

        try:
            # Use image_to_data to get word-level text and confidence scores
            data = pytesseract.image_to_data(
                processed_crop, 
                config=custom_config_psm7, 
                output_type=pytesseract.Output.DICT
            )
            text_psm7, confidence_psm7 = self._parse_ocr_data(data)

            # If confidence is low or text empty, try PSM 8 (single word)
            if confidence_psm7 < 50.0 or len(text_psm7.strip()) < 4:
                data_psm8 = pytesseract.image_to_data(
                    processed_crop, 
                    config=custom_config_psm8, 
                    output_type=pytesseract.Output.DICT
                )
                text_psm8, confidence_psm8 = self._parse_ocr_data(data_psm8)
                
                # Pick the result with higher confidence or longer text length
                if len(text_psm8) > len(text_psm7) and confidence_psm8 > 40.0:
                    text, confidence = text_psm8, confidence_psm8
                else:
                    text, confidence = text_psm7, confidence_psm7
            else:
                text, confidence = text_psm7, confidence_psm7
                
        except Exception as e:
            logging.error(f"Error executing Tesseract OCR: {e}")
            # Fallback to simple string extraction if image_to_data fails
            try:
                text = pytesseract.image_to_string(processed_crop, config=custom_config_psm7)
                confidence = 50.0  # arbitrary default
            except Exception as e_inner:
                logging.error(f"Fallback Tesseract OCR failed: {e_inner}")

        return text.strip(), confidence

    def _parse_ocr_data(self, ocr_data):
        """
        Parses the dictionary output of image_to_data to extract text and calculate average confidence.
        """
        words = []
        confidences = []

        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            try:
                conf = float(ocr_data['conf'][i])
            except (ValueError, TypeError):
                conf = -1.0

            # Filter out non-matching empty strings and confidences of -1
            if word and conf != -1:
                # If Tesseract successfully reads a word but reports 0 or less confidence,
                # assign a baseline confidence of 75.0 to represent a readable word.
                if conf <= 0.0:
                    conf = 75.0
                words.append(word)
                confidences.append(conf)

        if not words:
            return "", 0.0

        # Join words
        full_text = "".join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Heuristic: If confidence scores are normalized in range [0, 1], scale to [0, 100]
        if 0.0 < avg_confidence <= 1.0:
            avg_confidence *= 100.0

        return full_text, avg_confidence

if __name__ == "__main__":
    # Test OCR initialization
    ocr = OCREngine()

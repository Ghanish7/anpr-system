import os
import sys
import pytesseract

# Tesseract OCR settings
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Haar Cascade file for plate detection
HAAR_CASCADE_NAME = "haarcascade_russian_plate_number.xml"
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"

# Character validation mapping for common OCR errors
# Used to swap letters for numbers or numbers for letters in specific positions.
LETTER_TO_NUMBER = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'T': '7', 'J': '1', 'D': '0'
}

NUMBER_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '7': 'T'
}

# License plate regex patterns for validation
# 1. Indian plates: State Code (2 letters) + District Code (2 digits) + Unique Series (1 or 2 letters) + Number (4 digits)
#    Example: MH12DE1433, DL3CAY1102
# 2. General fallback pattern: alphanumeric between 4 and 15 characters
PLATE_PATTERNS = {
    "IN": r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$",
    "GENERAL": r"^[A-Z0-9]{4,15}$"
}

# CSV report output settings
CSV_OUTPUT_PATH = "anpr_results.csv"

# Known vehicle database path
VEHICLE_DB_PATH = "vehicle_db.json"

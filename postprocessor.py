import re
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PlatePostprocessor:
    def __init__(self):
        self.letter_to_number = config.LETTER_TO_NUMBER
        self.number_to_letter = config.NUMBER_TO_LETTER
        self.patterns = config.PLATE_PATTERNS

    def clean_text(self, text):
        """
        Removes spaces, punctuation, and non-alphanumeric characters,
        and converts the text to uppercase.
        """
        if not text:
            return ""
        # Convert to uppercase and strip
        text = text.upper().strip()
        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text)
        return cleaned

    def correct_character(self, char, expected_type):
        """
        Corrects a single character based on its expected type ('letter' or 'digit').
        """
        if expected_type == "letter":
            if char in self.number_to_letter:
                return self.number_to_letter[char]
        elif expected_type == "digit":
            if char in self.letter_to_number:
                return self.letter_to_number[char]
        return char

    def correct_indian_format(self, text):
        """
        Applies positional corrections specifically for Indian plates:
        Format is typically: State (2 letters) + District (2 digits) + Series (1-2 letters) + Unique No (4 digits)
        - Length 10: AA NN AA NNNN -> L L D D L L D D D D
        - Length 9:  AA NN A NNNN  -> L L D D L D D D D D
        Where L = Letter, D = Digit
        """
        if len(text) not in [9, 10]:
            # Can't reliably apply the standard positional template if length is different
            return text

        chars = list(text)
        
        # Positional template mappings (0-indexed)
        if len(text) == 10:
            template = ["letter", "letter", "digit", "digit", "letter", "letter", "digit", "digit", "digit", "digit"]
        else: # length 9
            template = ["letter", "letter", "digit", "digit", "letter", "digit", "digit", "digit", "digit"]

        corrected_chars = []
        for i, char in enumerate(chars):
            expected = template[i]
            corrected_chars.append(self.correct_character(char, expected))

        return "".join(corrected_chars)

    def validate_plate(self, text):
        """
        Validates the plate number against configured regex patterns.
        
        Returns:
        - match_type (str): The format it matched (e.g. 'IN', 'GENERAL'), or None if no match.
        """
        for format_name, pattern in self.patterns.items():
            if re.match(pattern, text):
                return format_name
        return None

    def process(self, raw_text):
        """
        Runs the complete post-processing pipeline on raw OCR text:
        1. Cleans the text (removes noise characters, converts to uppercase)
        2. Applies positional corrections (tries Indian plate logic, then fallback)
        3. Validates the output against formats
        
        Returns:
        - dict: Cleaned text, corrected text, validation status, and matching format name.
        """
        cleaned = self.clean_text(raw_text)
        if not cleaned:
            return {
                "raw": raw_text,
                "cleaned": "",
                "corrected": "",
                "valid": False,
                "format": None
            }

        # Apply corrections
        # 1. Try Indian plate correction if the text is close to length 9/10
        if len(cleaned) in [9, 10]:
            corrected = self.correct_indian_format(cleaned)
        else:
            corrected = cleaned

        # Validate the corrected plate
        matched_format = self.validate_plate(corrected)
        
        # If corrected is not valid but raw cleaned text is valid, prefer cleaned
        if not matched_format:
            cleaned_format = self.validate_plate(cleaned)
            if cleaned_format:
                corrected = cleaned
                matched_format = cleaned_format

        return {
            "raw": raw_text,
            "cleaned": cleaned,
            "corrected": corrected,
            "valid": matched_format is not None,
            "format": matched_format
        }

if __name__ == "__main__":
    # Test cases for post-processor
    post = PlatePostprocessor()
    
    # 1. Indian plate with O/0 and I/1 confusions
    test_1 = "MHl2DEl433" # 'l' instead of '1', 'l' instead of '1'
    result_1 = post.process(test_1)
    print(f"Raw: {test_1} -> Corrected: {result_1['corrected']} (Valid: {result_1['valid']}, Format: {result_1['format']})")
    
    # 2. Indian plate 9 chars with O/0 confusions
    test_2 = "DL03CAY11O2" # O at the end instead of 0
    # Wait, "DL03CAY11O2" is 11 chars? No, D-L-0-3-C-A-Y-1-1-O-2 is 11 chars. Let's trace:
    # DL (state) 03 (district) CAY (3 letters? In some states like DL, series can be 3 letters? e.g. DL 3C AY 1102. Yes, DL3CAY1102 is 10 chars. Let's count: D L 3 C A Y 1 1 0 2 -> 10 chars.
    # What about DL03CAY1102? D L 0 3 C A Y 1 1 0 2 -> 11 chars.
    test_3 = "DL3CAY11O2" # DL3CAY1102 with 'O' instead of '0'
    result_3 = post.process(test_3)
    print(f"Raw: {test_3} -> Corrected: {result_3['corrected']} (Valid: {result_3['valid']}, Format: {result_3['format']})")

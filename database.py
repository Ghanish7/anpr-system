import os
import json
import logging
import difflib
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VehicleDatabase:
    def __init__(self, db_path=None):
        """
        Initializes the vehicle database. Loads records from a JSON file.
        Creates a default database if it doesn't exist.
        """
        self.db_path = db_path if db_path is not None else config.VEHICLE_DB_PATH
        self.vehicles = {}
        self._load_or_initialize_db()

    def _load_or_initialize_db(self):
        """
        Loads the database or creates it with default sample records if missing.
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.vehicles = json.load(f)
                logging.info(f"Loaded {len(self.vehicles)} vehicle records from {self.db_path}.")
            except Exception as e:
                logging.error(f"Error loading vehicle database: {e}. Reinitializing default database...")
                self._create_default_db()
        else:
            self._create_default_db()

    def _create_default_db(self):
        """
        Populates a default dictionary of known vehicles and saves it to JSON.
        """
        self.vehicles = {
            "MH12DE1433": {
                "owner": "Rajesh Kumar",
                "make": "Maruti Suzuki",
                "model": "Swift",
                "color": "Silver",
                "status": "Active",
                "insurance_expiry": "2027-04-12"
            },
            "DL7CQ1939": {
                "owner": "Ghanish",
                "make": "Maybach",
                "model": "S-Class",
                "color": "Obsidian Black",
                "status": "Active",
                "insurance_expiry": "2028-09-15"
            },
            "DL3CAY1102": {
                "owner": "Anjali Sharma",
                "make": "Hyundai",
                "model": "i20",
                "color": "White",
                "status": "Active",
                "insurance_expiry": "2026-11-20"
            },
            "KA03MM5544": {
                "owner": "Vikram Singh",
                "make": "Honda",
                "model": "City",
                "color": "Black",
                "status": "Active",
                "insurance_expiry": "2027-08-01"
            },
            "HR26CT8899": {
                "owner": "Amit Chaudhary",
                "make": "Toyota",
                "model": "Fortuner",
                "color": "Grey",
                "status": "Suspended (Unpaid Fines)",
                "insurance_expiry": "2025-05-15"
            }
        }
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.vehicles, f, indent=4)
            logging.info(f"Created default vehicle database at {self.db_path}.")
        except Exception as e:
            logging.error(f"Error creating default vehicle database: {e}")

    def lookup_plate(self, plate_number, fuzzy_threshold=0.8):
        """
        Looks up a plate number in the database.
        Supports exact match first, and falls back to fuzzy match if exact is not found.
        
        Parameters:
        - plate_number (str): The cleaned plate number to lookup.
        - fuzzy_threshold (float): Similarity threshold (0.0 to 1.0) for fuzzy matching.
        
        Returns:
        - match_info (dict): A dictionary with keys 'match_type', 'plate', and 'details', or None.
        """
        if not plate_number:
            return None

        # Clean plate_number to match database keys
        plate_clean = plate_number.upper().replace(" ", "")

        # 1. Exact Match
        if plate_clean in self.vehicles:
            return {
                "match_type": "exact",
                "plate": plate_clean,
                "details": self.vehicles[plate_clean]
            }

        # 2. Fuzzy Match (helps handle minor OCR character misses)
        known_plates = list(self.vehicles.keys())
        matches = difflib.get_close_matches(plate_clean, known_plates, n=1, cutoff=fuzzy_threshold)
        
        if matches:
            best_match = matches[0]
            # Calculate similarity score
            similarity = difflib.SequenceMatcher(None, plate_clean, best_match).ratio()
            logging.info(f"Fuzzy match found: {best_match} for input {plate_clean} (similarity: {similarity:.2f})")
            return {
                "match_type": "fuzzy",
                "similarity": similarity,
                "plate": best_match,
                "details": self.vehicles[best_match]
            }

        return None

    def add_vehicle(self, plate_number, owner, make, model, color, status="Active", insurance_expiry="2027-01-01"):
        """
        Adds or updates a vehicle record.
        """
        plate_clean = plate_number.upper().replace(" ", "")
        self.vehicles[plate_clean] = {
            "owner": owner,
            "make": make,
            "model": model,
            "color": color,
            "status": status,
            "insurance_expiry": insurance_expiry
        }
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.vehicles, f, indent=4)
            logging.info(f"Added/Updated vehicle: {plate_clean}")
            return True
        except Exception as e:
            logging.error(f"Error saving vehicle record: {e}")
            return False

if __name__ == "__main__":
    db = VehicleDatabase()
    
    # Test exact match
    print("Exact Match:", db.lookup_plate("MH12DE1433"))
    
    # Test fuzzy match (with minor OCR mistake '1' replaced by 'I')
    print("Fuzzy Match:", db.lookup_plate("MH12DEI433"))
    
    # Test no match
    print("No Match:", db.lookup_plate("XX00XX0000"))

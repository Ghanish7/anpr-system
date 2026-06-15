import os
import cv2
import pandas as pd
import argparse
import logging
from datetime import datetime
import config
from detector import PlateDetector
from ocr_engine import OCREngine
from postprocessor import PlatePostprocessor
from database import VehicleDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ANPRPipeline:
    def __init__(self, tesseract_cmd=None, cascade_path=None, db_path=None):
        """
        Initializes the complete ANPR Pipeline with sub-modules.
        """
        # Override config variables if provided
        if tesseract_cmd:
            config.TESSERACT_CMD = tesseract_cmd
            
        self.detector = PlateDetector(cascade_path=cascade_path)
        self.ocr_engine = OCREngine()
        self.postprocessor = PlatePostprocessor()
        self.db = VehicleDatabase(db_path=db_path)

    def log_to_csv(self, csv_path, filename, plate_number, confidence, owner, status, match_type):
        """
        Saves ANPR results to a CSV file.
        Appends to existing file or creates a new one with header.
        Columns: filename, plate_number, confidence, timestamp, owner, status, match_type
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "filename": [filename],
            "plate_number": [plate_number],
            "confidence": [round(confidence, 2)],
            "timestamp": [timestamp],
            "owner": [owner if owner else "Unknown"],
            "vehicle_status": [status if status else "N/A"],
            "db_match_type": [match_type if match_type else "None"]
        }
        df = pd.DataFrame(record)
        
        try:
            # If CSV file exists, append without header, else write with header
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, mode="w", header=True, index=False)
            logging.info(f"Result appended to CSV: {csv_path}")
        except Exception as e:
            logging.error(f"Error saving result to CSV: {e}")

    def process_image(self, image_path, output_csv=config.CSV_OUTPUT_PATH, visualize=True, save_annotated=True, fallback_to_full_image=True):
        """
        Executes the full pipeline on a single image.
        1. Loads image.
        2. Detects plates.
        3. Crops, preprocesses, and OCRs plate regions.
        4. Cleans and validates plate numbers.
        5. Performs database query.
        6. Annotates visual results.
        7. Saves logs and displays/saves output images.
        """
        if not os.path.exists(image_path):
            logging.error(f"Image not found at: {image_path}")
            return None

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image from {image_path}")
            return None

        filename = os.path.basename(image_path)
        logging.info(f"\n--- Processing Image: {filename} ---")

        # 1. Detect plates
        detections = self.detector.detect_plates(img)
        
        if not detections:
            if fallback_to_full_image:
                logging.warning("No license plates detected. Falling back to treating the entire image as the plate crop.")
                h, w = img.shape[:2]
                detections = [{
                    "box": (0, 0, w, h),
                    "crop": img
                }]
            else:
                logging.warning("No license plates detected in the image.")
                # Log empty plate detection to CSV
                self.log_to_csv(
                    csv_path=output_csv,
                    filename=filename,
                    plate_number="NO_PLATE_DETECTED",
                    confidence=0.0,
                    owner="N/A",
                    status="N/A",
                    match_type="None"
                )
                return img

        processed_detections = []
        
        for idx, det in enumerate(detections):
            box = det["box"]
            crop = det["crop"]
            
            # 2. OCR OCR Module
            raw_text, ocr_conf = self.ocr_engine.recognize_text(crop)
            logging.info(f"Plate {idx+1} - Raw OCR Text: '{raw_text}' | Confidence: {ocr_conf:.1f}%")
            
            # 3. Post-processing
            post_result = self.postprocessor.process(raw_text)
            plate_number = post_result["corrected"]
            logging.info(f"Plate {idx+1} - Post-Processed: '{plate_number}' | Valid format: {post_result['valid']}")
            
            # 4. Database query
            db_result = self.db.lookup_plate(plate_number)
            owner, vehicle_status, match_type = None, None, None
            
            if db_result:
                owner = db_result["details"]["owner"]
                vehicle_status = db_result["details"]["status"]
                match_type = db_result["match_type"]
                vehicle_desc = f"{db_result['details']['make']} {db_result['details']['model']}"
                logging.info(f"Plate {idx+1} - MATCH FOUND: {owner} ({vehicle_desc}) | Status: {vehicle_status}")
            else:
                logging.info(f"Plate {idx+1} - No database record found.")
                
            # Log to CSV
            self.log_to_csv(
                csv_path=output_csv,
                filename=filename,
                plate_number=plate_number if plate_number else "OCR_FAILED",
                confidence=ocr_conf,
                owner=owner,
                status=vehicle_status,
                match_type=match_type
            )
            
            # Add text and confidence to annotation metadata
            det_label = plate_number if plate_number else "Plate"
            processed_detections.append({
                "box": box,
                "text": det_label,
                "confidence": ocr_conf
            })

            # Print console summary
            print(f"\n[Result] File: {filename}")
            print(f"  +- Detected Plate: {plate_number}")
            print(f"  +- OCR Confidence: {ocr_conf:.2f}%")
            print(f"  +- Database Record: {owner if owner else 'Not Found'} ({vehicle_status if vehicle_status else 'N/A'})")

        # 5. Annotate Image
        annotated_img = self.detector.draw_annotations(img, processed_detections)

        # 6. Save annotated image
        if save_annotated:
            out_dir = os.path.join(os.path.dirname(image_path) or ".", "results")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"annotated_{filename}")
            cv2.imwrite(save_path, annotated_img)
            logging.info(f"Annotated image saved to: {save_path}")

        # 7. Visualize (GUI Window)
        if visualize:
            try:
                cv2.imshow("ANPR Result", annotated_img)
                logging.info("Press any key in the window to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                logging.warning(
                    f"GUI visualization failed (this is expected in headless or remote environments): {e}. "
                    f"Falling back to saved file."
                )

        return annotated_img

    def process_batch(self, directory_path, output_csv=config.CSV_OUTPUT_PATH, visualize=False, save_annotated=True):
        """
        Executes pipeline in batch mode for all images in a folder.
        Force-disables interactive visual GUI popups for batch processing by default.
        """
        if not os.path.exists(directory_path):
            logging.error(f"Directory not found: {directory_path}")
            return
            
        supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.lower().endswith(supported_extensions)
        ]
        
        if not files:
            logging.warning(f"No supported images found in directory: {directory_path}")
            return

        logging.info(f"Starting batch processing of {len(files)} images...")
        for file in files:
            try:
                self.process_image(
                    image_path=file,
                    output_csv=output_csv,
                    visualize=visualize,
                    save_annotated=save_annotated
                )
            except Exception as e:
                logging.error(f"Failed to process {file}: {e}", exc_info=True)
                
        logging.info("Batch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Automatic Number Plate Recognition (ANPR) System")
    parser.add_argument("--image", type=str, help="Path to a single input image")
    parser.add_argument("--batch", type=str, help="Path to a directory of images for batch processing")
    parser.add_argument("--output", type=str, default=config.CSV_OUTPUT_PATH, help="Path to save CSV output logs")
    parser.add_argument("--no-visualize", action="store_true", help="Disable OpenCV popup windows")
    parser.add_argument("--no-save-annotated", action="store_true", help="Disable saving annotated output images")
    parser.add_argument("--tesseract-path", type=str, help="Manually override Tesseract executable path")
    parser.add_argument("--cascade", type=str, help="Manually override Haar Cascade XML file path")
    
    args = parser.parse_args()

    # Initialize the pipeline
    try:
        pipeline = ANPRPipeline(
            tesseract_cmd=args.tesseract_path, 
            cascade_path=args.cascade
        )
        
        visualize = not args.no_visualize
        save_annotated = not args.no_save_annotated

        if args.image:
            pipeline.process_image(
                image_path=args.image,
                output_csv=args.output,
                visualize=visualize,
                save_annotated=save_annotated
            )
        elif args.batch:
            # Batch mode disables interactive visualization by default to prevent pausing
            pipeline.process_batch(
                directory_path=args.batch,
                output_csv=args.output,
                visualize=False,
                save_annotated=save_annotated
            )
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"Pipeline error: {e}", exc_info=True)

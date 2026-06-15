from flask import Flask, request, jsonify, render_template
import os
import cv2
import pandas as pd
from main import ANPRPipeline
import config

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'results'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)

# Initialize pipeline
pipeline = ANPRPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Run detection
        detections = pipeline.detector.detect_plates(img)
        
        fallback = False
        if not detections:
            # Fallback to full image crop
            h, w = img.shape[:2]
            detections = [{
                "box": (0, 0, w, h),
                "crop": img
            }]
            fallback = True
            
        results = []
        processed_detections = []
        
        for idx, det in enumerate(detections):
            box = det["box"]
            crop = det["crop"]
            
            # Save cropped plate image for display
            crop_filename = f"crop_{idx}_{filename}"
            crop_path = os.path.join(app.root_path, 'static', 'results', crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # OCR and post-processing
            raw_text, ocr_conf = pipeline.ocr_engine.recognize_text(crop)
            post_result = pipeline.postprocessor.process(raw_text)
            plate_number = post_result["corrected"]
            
            # Database query
            db_result = pipeline.db.lookup_plate(plate_number)
            owner, vehicle_status, make, model = None, None, None, None
            match_type = "None"
            
            if db_result:
                owner = db_result["details"]["owner"]
                vehicle_status = db_result["details"]["status"]
                make = db_result["details"]["make"]
                model = db_result["details"]["model"]
                match_type = db_result["match_type"]
                
            # Log to CSV
            pipeline.log_to_csv(
                csv_path=config.CSV_OUTPUT_PATH,
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
            
            results.append({
                "plate_number": plate_number if plate_number else "OCR_FAILED",
                "confidence": round(ocr_conf, 2),
                "valid_format": post_result["valid"],
                "crop_url": f"/static/results/{crop_filename}",
                "owner": owner if owner else "Unknown",
                "vehicle_status": vehicle_status if vehicle_status else "N/A",
                "make_model": f"{make} {model}" if make else "Unknown",
                "db_match_type": match_type
            })
            
        # Draw and save annotated original image
        annotated_img = pipeline.detector.draw_annotations(img, processed_detections)
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.root_path, 'static', 'results', annotated_filename)
        cv2.imwrite(annotated_path, annotated_img)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "annotated_url": f"/static/results/{annotated_filename}",
            "fallback_used": fallback,
            "plates": results
        })

@app.route('/history', methods=['GET'])
def get_history():
    if os.path.exists(config.CSV_OUTPUT_PATH):
        try:
            df = pd.read_csv(config.CSV_OUTPUT_PATH)
            df = df.fillna("")
            # Get last 15 scans, reverse to show newest first
            records = df.tail(15).to_dict(orient='records')
            records.reverse()
            return jsonify(records)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

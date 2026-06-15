# Automatic Number Plate Recognition (ANPR / LPR) System

A modular, clean, and highly customizable Python system for detecting vehicle license plates and recognizing characters using OpenCV and Tesseract OCR.

---

## Features

1. **License Plate Detection**: Uses OpenCV's Cascade Classifier (Haar Cascades) to detect license plates. Downloads pre-trained models automatically.
2. **Text Recognition (OCR)**: Uses Tesseract OCR (`pytesseract`) configured with optimized flags (`--psm 7` / `--psm 8`) and character whitelisting for license plates.
3. **Advanced Preprocessing**: Grayscaling, cubic resizing, bilateral filtering (denoising while keeping borders sharp), Otsu's binarization, and border-based background inversion.
4. **Post-Processing & Correction**:
   - Removes spaces and special characters.
   - Positional correction (e.g., swapping `O` to `0` or `1` to `I` based on their character indices).
   - Regex-based format validation (specifically for Indian state-rto formats and general formats).
5. **Vehicle Database Lookup**: Exact and fuzzy match querying against a local database (`vehicle_db.json`) of registered owners and vehicle statuses.
6. **Reporting & Logging**: Outputs CSV records (`filename`, `plate_number`, `confidence`, `timestamp`, `owner`, `vehicle_status`, `db_match_type`) and saves annotated result images with bounding boxes.

---

## Directory Structure

```text
3345-project/
├── requirements.txt         # Python library dependencies
├── config.py                # System paths, character mappings, and regex configurations
├── detector.py              # Plate detector logic and cascade downloader
├── ocr_engine.py            # Image preprocessing pipeline and pytesseract runner
├── postprocessor.py         # Cleaning, positional corrections, and format validation
├── database.py              # Register lookup and fuzzy match database
├── main.py                  # Integration pipeline (CLI endpoint for single/batch files)
├── test_pipeline.py         # Automated self-contained synthetic test script
└── README.md                # System documentation
```

---

## Setup Instructions

### 1. Install Python Packages
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (System Dependency)
Tesseract must be installed on your operating system for character recognition to work.

#### **Windows**
1. Download the installer from the official UB Mannheim build repository:
   [Tesseract Installer for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer. Default path is typically: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. If installed in a different location, open [config.py](file:///c:/Users/ECS%20LAB3/Documents/3345-project/config.py) and update `TESSERACT_CMD`:
   ```python
   TESSERACT_CMD = r"D:\Path\To\Tesseract-OCR\tesseract.exe"
   ```

#### **Linux (Ubuntu/Debian)**
Run the following package install command:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### **macOS**
Install via Homebrew:
```bash
brew install tesseract
```

---

## Usage Guide

### 1. Run Automated Verification Test
Run the test script to verify that the pipeline is working. It generates a local synthetic license plate image and executes the complete pipeline:
```bash
python test_pipeline.py
```
*Expected Output*: Displays step-by-step logs, prints test coverage, writes results to `test_results.csv`, and saves annotated crop results in the `./results/` folder.

### 2. Single Image Processing
To run the ANPR pipeline on a single vehicle image:
```bash
python main.py --image path/to/your/car.jpg
```
*Note*: Adding `--no-visualize` runs the system in headless mode (doesn't open a popup window), saving the output under `results/annotated_car.jpg`.

### 3. Batch Directory Processing
To process a folder of vehicle images:
```bash
python main.py --batch path/to/image_folder --output batch_results.csv
```

---

## Positional Correction Example

When recognizing a plate number (like `MH12DE1433`), OCR engines often mistake `1` (one) for `I` (capital eye) or `0` (zero) for `O` (capital oh). 

Our post-processor automatically detects the pattern template (Letters vs. Digits) and maps character swaps:
*   **Raw OCR Output**: `MHl2DEl433`
*   **Post-Processed Output**: `MH12DE1433` (Automatically corrected using index rules and letter/number dictionaries)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import easyocr
import numpy as np
import io
import difflib
from typing import Optional
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI(
    title="OCR Text Extraction API",
    description="API for extracting text from images using EasyOCR and TrOCR",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
easyocr_reader = None
trocr_processor = None
trocr_model = None

def load_easyocr():
    """Load EasyOCR reader"""
    global easyocr_reader
    if easyocr_reader is None:
        easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return easyocr_reader

def load_trocr():
    """Load TrOCR models"""
    global trocr_processor, trocr_model
    if trocr_processor is None or trocr_model is None:
        try:
            trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            trocr_model.eval()
        except Exception as e:
            print(f"Error loading TrOCR: {e}")
    return trocr_processor, trocr_model

def calculate_similarity(extracted_text: str, expected_text: str):
    """Calculate text similarity"""
    extracted_clean = extracted_text.strip()
    expected_clean = expected_text.strip()
    
    total_chars_expected = len(expected_clean)
    total_chars_extracted = len(extracted_clean)
    
    matches = 0
    diff = list(difflib.ndiff(expected_clean, extracted_clean))
    
    for item in diff:
        if item.startswith('  '):
            matches += 1
    
    if total_chars_expected == 0:
        similarity_score = 1.0 if total_chars_extracted == 0 else 0.0
    else:
        similarity_score = matches / max(total_chars_expected, total_chars_extracted)
    
    return {
        'similarity_score': similarity_score * 100,
        'matches': matches,
        'total_expected': total_chars_expected,
        'total_extracted': total_chars_extracted,
        'character_accuracy': (matches / total_chars_expected) * 100 if total_chars_expected > 0 else 0
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OCR Text Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "extract_text": "/api/extract",
            "verify_text": "/api/verify",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "easyocr_loaded": easyocr_reader is not None,
        "trocr_loaded": trocr_processor is not None and trocr_model is not None
    }

@app.post("/api/extract")
async def extract_text(
    file: UploadFile = File(...),
    text_type: str = Form(default="printed"),
    min_confidence: float = Form(default=0.1)
):
    """
    Extract text from an uploaded image
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG)
    - text_type: 'printed' or 'handwritten'
    - min_confidence: Minimum confidence threshold (0.0 to 1.0)
    
    Returns:
    - extracted_text: Complete extracted text
    - regions: List of detected text regions with coordinates and confidence
    - processing_time: Time taken to process the image
    """
    try:
        import time
        start_time = time.time()
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Load EasyOCR
        reader = load_easyocr()
        
        # Process based on text type
        if text_type.lower() == "handwritten":
            processor, model = load_trocr()
            if processor is None or model is None:
                raise HTTPException(status_code=500, detail="TrOCR models not available")
            
            # Detect regions with EasyOCR
            easyocr_results = reader.readtext(image_np)
            
            results = []
            for bbox, _, confidence in easyocr_results:
                # Extract region coordinates
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Crop and process with TrOCR
                cropped = image.crop((x_min, y_min, x_max, y_max))
                pixel_values = processor(cropped, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                results.append((bbox, text, confidence))
        else:
            # Standard EasyOCR for printed text
            results = reader.readtext(image_np)
        
        # Filter by confidence
        filtered_results = [
            (bbox, text, prob) for bbox, text, prob in results 
            if prob >= min_confidence
        ]
        
        # Format response
        regions = []
        for bbox, text, confidence in filtered_results:
            regions.append({
                "text": text,
                "confidence": float(confidence),
                "bounding_box": {
                    "top_left": [float(bbox[0][0]), float(bbox[0][1])],
                    "top_right": [float(bbox[1][0]), float(bbox[1][1])],
                    "bottom_right": [float(bbox[2][0]), float(bbox[2][1])],
                    "bottom_left": [float(bbox[3][0]), float(bbox[3][1])]
                }
            })
        
        extracted_text = " ".join([r["text"] for r in regions])
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "extracted_text": extracted_text,
            "total_regions": len(regions),
            "regions": regions,
            "processing_time": round(processing_time, 2),
            "text_type": text_type
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify")
async def verify_text(
    file: UploadFile = File(...),
    expected_text: str = Form(...),
    text_type: str = Form(default="printed"),
    min_confidence: float = Form(default=0.1)
):
    """
    Extract text from image and verify against expected text
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG)
    - expected_text: Expected text to compare against
    - text_type: 'printed' or 'handwritten'
    - min_confidence: Minimum confidence threshold (0.0 to 1.0)
    
    Returns:
    - extracted_text: Text extracted from image
    - expected_text: Text provided for comparison
    - similarity_score: Percentage similarity (0-100)
    - character_accuracy: Accuracy of character matching
    - verification_result: 'excellent', 'good', 'fair', or 'poor'
    """
    try:
        import time
        start_time = time.time()
        
        # First extract text using the extract endpoint logic
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        reader = load_easyocr()
        
        if text_type.lower() == "handwritten":
            processor, model = load_trocr()
            if processor is None or model is None:
                raise HTTPException(status_code=500, detail="TrOCR models not available")
            
            easyocr_results = reader.readtext(image_np)
            results = []
            for bbox, _, confidence in easyocr_results:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                cropped = image.crop((x_min, y_min, x_max, y_max))
                pixel_values = processor(cropped, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                results.append((bbox, text, confidence))
        else:
            results = reader.readtext(image_np)
        
        filtered_results = [
            (bbox, text, prob) for bbox, text, prob in results 
            if prob >= min_confidence
        ]
        
        extracted_text = " ".join([text for _, text, _ in filtered_results])
        
        # Calculate similarity
        similarity_metrics = calculate_similarity(extracted_text, expected_text)
        
        # Determine verification result
        score = similarity_metrics['similarity_score']
        if score >= 90:
            verification_result = "excellent"
        elif score >= 75:
            verification_result = "good"
        elif score >= 50:
            verification_result = "fair"
        else:
            verification_result = "poor"
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "extracted_text": extracted_text,
            "expected_text": expected_text,
            "similarity_score": round(similarity_metrics['similarity_score'], 2),
            "character_accuracy": round(similarity_metrics['character_accuracy'], 2),
            "matches": similarity_metrics['matches'],
            "total_expected_chars": similarity_metrics['total_expected'],
            "total_extracted_chars": similarity_metrics['total_extracted'],
            "verification_result": verification_result,
            "processing_time": round(processing_time, 2),
            "text_type": text_type
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

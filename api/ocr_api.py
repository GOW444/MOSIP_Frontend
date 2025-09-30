from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import List, Dict, Any
import tempfile
import os

# Initialize FastAPI app
app = FastAPI(
    title="OCR Text Extraction API",
    description="Extract text from images using EasyOCR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader (cached globally)
reader = easyocr.Reader(['en'], gpu=False)

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "OCR Text Extraction API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "supported_languages": reader.lang_list
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "OCR Text Extraction API",
        "supported_languages": reader.lang_list,
        "supported_formats": ["PNG", "JPG", "JPEG"],
        "version": "1.0.0"
    }

@app.post("/api/v1/extract-text", tags=["OCR"])
async def extract_text(
    file: UploadFile = File(...),
    min_confidence: float = 0.1
):
    """
    Extract text from uploaded image file
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG)
    - min_confidence: Minimum confidence threshold (0.0 to 1.0)
    
    Returns:
    - JSON response with extracted text and metadata
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file (PNG, JPG, JPEG)."
        )
    
    # Validate confidence threshold
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="min_confidence must be between 0.0 and 1.0"
        )
    
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Convert to PIL Image
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image_pil)
        
        # Run OCR
        results = reader.readtext(image_np)
        
        # Filter results by confidence threshold
        filtered_results = [
            (bbox, text, prob) for bbox, text, prob in results 
            if prob >= min_confidence
        ]
        
        # Process results
        extracted_data = []
        full_text = ""
        
        for i, (bbox, text, confidence) in enumerate(filtered_results):
            extracted_data.append({
                "region_id": i + 1,
                "text": text,
                "confidence": round(float(confidence), 3),
                "bounding_box": {
                    "top_left": [int(bbox[0][0]), int(bbox[0][1])],
                    "top_right": [int(bbox[1][0]), int(bbox[1][1])],
                    "bottom_right": [int(bbox[2][0]), int(bbox[2][1])],
                    "bottom_left": [int(bbox[3][0]), int(bbox[3][1])]
                }
            })
            full_text += text + " "
        
        # Calculate statistics
        total_regions = len(filtered_results)
        avg_confidence = (
            sum(prob for _, _, prob in filtered_results) / total_regions
            if total_regions > 0 else 0
        )
        total_characters = sum(len(text) for _, text, _ in filtered_results)
        
        # Prepare response
        response_data = {
            "status": "success",
            "message": "Text extraction completed successfully",
            "data": {
                "extracted_text": full_text.strip(),
                "regions": extracted_data,
                "statistics": {
                    "total_regions": total_regions,
                    "average_confidence": round(avg_confidence, 3),
                    "total_characters": total_characters,
                    "min_confidence_used": min_confidence
                },
                "metadata": {
                    "filename": file.filename,
                    "file_size_bytes": len(image_bytes),
                    "image_dimensions": {
                        "width": image_pil.width,
                        "height": image_pil.height
                    },
                    "supported_languages": reader.lang_list
                }
            }
        }
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )

@app.get("/api/v1/supported-languages", tags=["Info"])
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": reader.lang_list,
        "default_language": "en"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

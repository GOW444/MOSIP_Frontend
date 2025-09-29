# 🚀 MOSIP OCR API - Streamlit Integration Guide

## 📋 API Base Information
- **Base URL**: `https://deandra-creamiest-unpenetratingly.ngrok-free.dev`
- **API Documentation**: `https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/docs`
- **API Version**: v1
- **Format**: JSON
- **Public Access**: Available worldwide via ngrok tunnel
- **⚠️ Important**: First-time visitors may see an ngrok warning page - click "Visit Site" to proceed

---

## 🔌 Available Endpoints

### 1. **Health Check**
```http
GET /
GET /health
```
**Response:**
```json
{
  "service": "MOSIP OCR API",
  "status": "running",
  "version": "1.0.0",
  "timestamp": "2025-09-29T22:35:59.123456",
  "endpoints": {
    "docs": "/api/docs",
    "health": "/health",
    "ocr_extract": "/api/v1/ocr/extract",
    "ocr_validate": "/api/v1/ocr/validate",
    "document_process": "/api/v1/document/process"
  }
}
```

---

### 2. **📸 Extract Text from Image** (Main OCR Endpoint)
```http
POST /api/v1/ocr/extract
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Image file (JPG, PNG, TIFF)
- `confidence_threshold` (optional): Float 0.0-1.0, default 0.7
- `preprocess` (optional): Boolean, default true
- `languages` (optional): String "en,hi,ta", default "en"

**Streamlit Example:**
```python
import streamlit as st
import requests

# Note: Add ngrok-skip-browser-warning header for API calls
headers = {"ngrok-skip-browser-warning": "true"}

# File uploader
uploaded_file = st.file_uploader("Upload Document Image", type=['jpg', 'jpeg', 'png', 'tiff'])

if uploaded_file:
    # Confidence slider
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    
    # Language selection
    languages = st.multiselect("Languages", ["en", "hi", "ta", "te"], default=["en"])
    
    if st.button("Extract Text"):
        # Prepare the request
        files = {"file": uploaded_file.getvalue()}
        data = {
            "confidence_threshold": confidence,
            "preprocess": True,
            "languages": ",".join(languages)
        }
        
        # Make API call with ngrok header
        response = requests.post(
            "https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/v1/ocr/extract",
            files={"file": uploaded_file},
            data=data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.success(f"✅ Found {result['text_blocks_found']} text blocks")
            st.write(f"⏱️ Processing time: {result['processing_time']}s")
            st.write(f"🎯 Average confidence: {result['average_confidence']}")
            
            # Show combined text
            st.subheader("📝 Extracted Text")
            st.text_area("Combined Text", result['combined_text'], height=100)
            
            # Show individual blocks
            st.subheader("📋 Text Blocks")
            for i, block in enumerate(result['text_blocks']):
                with st.expander(f"Block {i+1} (Confidence: {block['confidence']:.3f})"):
                    st.text(block['text'])
                    st.json(block['bbox'])
        else:
            st.error(f"❌ Error: {response.text}")
```

**Response:**
```json
{
  "status": "success",
  "processing_time": 3.456,
  "text_blocks_found": 4,
  "text_blocks": [
    {
      "text": "MOSIPOCR TEST",
      "confidence": 0.689,
      "bbox": [[20, 30], [180, 30], [180, 60], [20, 60]],
      "language": "en"
    },
    {
      "text": "Aadhaar: 1234 5678 9012",
      "confidence": 0.752,
      "bbox": [[20, 70], [250, 70], [250, 100], [20, 100]],
      "language": "en"
    }
  ],
  "combined_text": "MOSIPOCR TEST Aadhaar: 1234 5678 9012 PAN: ABCDE1234F Phone: 9876543210",
  "average_confidence": 0.756,
  "parameters": {
    "confidence_threshold": 0.7,
    "preprocess": true,
    "languages": ["en"]
  }
}
```

---

### 3. **✅ Validate Text**
```http
POST /api/v1/ocr/validate
Content-Type: application/json
```

**Streamlit Example:**
```python
import streamlit as st
import requests
import json

st.subheader("🔍 Text Validation")

# Text input
text_to_validate = st.text_area("Enter text to validate:")
document_type = st.selectbox("Document Type (optional)", 
                           ["", "aadhaar", "pan", "passport", "driving_license"])

if st.button("Validate Text") and text_to_validate:
    payload = {
        "text": text_to_validate,
        "document_type": document_type if document_type else None
    }
    
    response = requests.post(
        "https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/v1/ocr/validate",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result['is_valid']:
            st.success("✅ Text is valid!")
        else:
            st.error("❌ Text validation failed")
        
        st.write(f"📊 Rules passed: {result['rules_passed']}")
        st.write(f"📊 Rules failed: {result['rules_failed']}")
        st.write(f"💬 Message: {result['validation_message']}")
        
        # Show detailed results
        with st.expander("📋 Detailed Rules"):
            for rule in result['rule_details']:
                icon = "✅" if rule['valid'] else "❌"
                st.write(f"{icon} {rule['rule']}: {rule['message']}")
    else:
        st.error(f"❌ Error: {response.text}")
```

**Request:**
```json
{
  "text": "Aadhaar: 1234 5678 9012",
  "document_type": "aadhaar",
  "field_type": null
}
```

**Response:**
```json
{
  "status": "success",
  "text": "Aadhaar: 1234 5678 9012",
  "is_valid": true,
  "validation_message": "Validation passed: 2 passed, 0 failed",
  "rules_passed": 2,
  "rules_failed": 0,
  "rule_details": [
    {
      "rule": "length",
      "valid": true,
      "message": "Text length: 21",
      "details": {"length": 21, "min": 2, "max": 1000}
    },
    {
      "rule": "characters",
      "valid": true,
      "message": "All characters are valid",
      "details": {"invalid_chars": []}
    }
  ]
}
```

---

### 4. **🔄 Complete Document Processing** (Extract + Validate)
```http
POST /api/v1/document/process
Content-Type: multipart/form-data
```

**Streamlit Example:**
```python
import streamlit as st
import requests

st.subheader("📄 Complete Document Processing")

uploaded_file = st.file_uploader("Upload Document", type=['jpg', 'jpeg', 'png', 'tiff'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
        document_type = st.selectbox("Document Type", 
                                   ["", "aadhaar", "pan", "passport", "driving_license"])
    
    with col2:
        validate_fields = st.checkbox("Validate Fields", value=True)
    
    if st.button("Process Document"):
        files = {"file": uploaded_file}
        data = {
            "confidence_threshold": confidence,
            "document_type": document_type if document_type else None,
            "validate_fields": validate_fields
        }
        
        response = requests.post(
            "https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/v1/document/process",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Summary
            summary = result['summary']
            st.success(f"✅ Processing completed in {result['processing_time']}s")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Blocks", summary['total_blocks'])
            with col2:
                st.metric("Valid Blocks", summary['valid_blocks'])
            with col3:
                st.metric("Avg Confidence", f"{summary['average_confidence']:.3f}")
            
            # Combined text
            st.subheader("📝 Extracted Text")
            st.text_area("Result", summary['combined_text'], height=100)
            
            # Detailed blocks
            st.subheader("📋 Detailed Results")
            for i, block in enumerate(result['text_blocks']):
                with st.expander(f"Block {i+1}: {block['text'][:50]}..."):
                    st.write(f"**Text:** {block['text']}")
                    st.write(f"**Confidence:** {block['confidence']:.3f}")
                    
                    if 'validation' in block:
                        validation = block['validation']
                        if validation['is_valid']:
                            st.success("✅ Valid")
                        else:
                            st.error("❌ Invalid")
                        st.write(f"**Rules:** {validation['rules_passed']} passed")
        else:
            st.error(f"❌ Error: {response.text}")
```

---

### 5. **🌐 Get Supported Languages**
```http
GET /api/v1/languages
```

**Streamlit Example:**
```python
import streamlit as st
import requests

if st.button("Get Supported Languages"):
    response = requests.get("https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/v1/languages")
    
    if response.status_code == 200:
        result = response.json()
        
        st.success(f"✅ {result['total_count']} languages supported")
        st.write(f"**Default:** {result['default']}")
        
        # Display languages
        languages = result['supported_languages']
        for code, name in languages.items():
            st.write(f"- **{code}**: {name}")
```

**Response:**
```json
{
  "status": "success",
  "supported_languages": {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu"
  },
  "default": "en",
  "total_count": 12
}
```

---

## 🎯 Complete Streamlit App Example

```python
import streamlit as st
import requests
import json
from PIL import Image
import io

st.set_page_config(page_title="MOSIP OCR", page_icon="📄", layout="wide")

st.title("📄 MOSIP OCR - Text Extraction & Verification")
st.markdown("Upload document images to extract and validate text content")

# API Base URL
API_BASE = "https://deandra-creamiest-unpenetratingly.ngrok-free.dev"

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
preprocess = st.sidebar.checkbox("Image Preprocessing", value=True)

# Language selection
languages_response = requests.get(f"{API_BASE}/api/v1/languages")
if languages_response.status_code == 200:
    lang_data = languages_response.json()
    available_languages = list(lang_data['supported_languages'].keys())
    selected_languages = st.sidebar.multiselect(
        "Languages", 
        available_languages, 
        default=["en"]
    )
else:
    selected_languages = ["en"]

# Main interface
uploaded_file = st.file_uploader(
    "Upload Document Image", 
    type=['jpg', 'jpeg', 'png', 'tiff'],
    help="Upload an image file containing text to extract"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Document type selection
        document_type = st.selectbox(
            "Document Type (optional)",
            ["", "aadhaar", "pan", "passport", "driving_license", "voter_id"]
        )
    
    with col2:
        tab1, tab2, tab3 = st.tabs(["🔍 Extract Text", "✅ Validate", "🔄 Full Process"])
        
        with tab1:
            if st.button("Extract Text", key="extract"):
                with st.spinner("Extracting text..."):
                    files = {"file": uploaded_file.getvalue()}
                    data = {
                        "confidence_threshold": confidence_threshold,
                        "preprocess": preprocess,
                        "languages": ",".join(selected_languages)
                    }
                    
                    response = requests.post(
                        f"{API_BASE}/api/v1/ocr/extract",
                        files={"file": uploaded_file},
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Text Blocks", result['text_blocks_found'])
                        with col_b:
                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                        with col_c:
                            st.metric("Avg Confidence", f"{result['average_confidence']:.3f}")
                        
                        # Combined text
                        st.text_area("📝 Extracted Text", result['combined_text'], height=150)
                        
                        # Individual blocks
                        for i, block in enumerate(result['text_blocks']):
                            with st.expander(f"Block {i+1} (Confidence: {block['confidence']:.3f})"):
                                st.text(block['text'])
                                st.json({"bbox": block['bbox'], "language": block['language']})
                    else:
                        st.error(f"Error: {response.text}")
        
        with tab2:
            text_input = st.text_area("Text to validate:", height=100)
            
            if st.button("Validate Text", key="validate") and text_input:
                payload = {
                    "text": text_input,
                    "document_type": document_type if document_type else None
                }
                
                response = requests.post(
                    f"{API_BASE}/api/v1/ocr/validate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['is_valid']:
                        st.success("✅ Text is valid!")
                    else:
                        st.error("❌ Text validation failed")
                    
                    st.info(f"Rules passed: {result['rules_passed']}/{result['rules_passed'] + result['rules_failed']}")
                    
                    for rule in result['rule_details']:
                        icon = "✅" if rule['valid'] else "❌"
                        st.write(f"{icon} **{rule['rule']}**: {rule['message']}")
                else:
                    st.error(f"Error: {response.text}")
        
        with tab3:
            validate_fields = st.checkbox("Validate extracted fields", value=True)
            
            if st.button("Process Complete Document", key="process"):
                with st.spinner("Processing document..."):
                    files = {"file": uploaded_file.getvalue()}
                    data = {
                        "confidence_threshold": confidence_threshold,
                        "document_type": document_type if document_type else None,
                        "validate_fields": validate_fields
                    }
                    
                    response = requests.post(
                        f"{API_BASE}/api/v1/document/process",
                        files={"file": uploaded_file},
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        summary = result['summary']
                        
                        # Summary metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Total Blocks", summary['total_blocks'])
                        with col_b:
                            st.metric("Valid Blocks", summary['valid_blocks'])
                        with col_c:
                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                        with col_d:
                            st.metric("Avg Confidence", f"{summary['average_confidence']:.3f}")
                        
                        # Final text
                        st.text_area("📝 Validated Text", summary['combined_text'], height=120)
                        
                        # Detailed results
                        st.subheader("📋 Detailed Results")
                        for i, block in enumerate(result['text_blocks']):
                            with st.expander(f"Block {i+1}: {block['text'][:50]}..."):
                                st.text(f"Text: {block['text']}")
                                st.text(f"Confidence: {block['confidence']:.3f}")
                                
                                if 'validation' in block:
                                    val = block['validation']
                                    if val['is_valid']:
                                        st.success("✅ Valid")
                                    else:
                                        st.error("❌ Invalid")
                                    st.text(f"Rules: {val['rules_passed']} passed, {val['rules_failed']} failed")
                    else:
                        st.error(f"Error: {response.text}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MOSIP OCR API v1.0**")
st.sidebar.markdown("🚀 Powered by EasyOCR")
```

---

## ⚡ Quick Start Commands

**1. Start the API:**
```bash
cd /home/pilot/Desktop/MOSIP
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**2. Test API:**
```bash
curl https://deandra-creamiest-unpenetratingly.ngrok-free.dev/health
```

**3. View API Docs:**
Open: `https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/docs`

---

## 🔧 Error Handling

All endpoints return errors in this format:
```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid file, parameters)
- `422`: Validation error
- `500`: Internal server error

---

## 🎯 Performance Notes

- **Processing Time**: 2-5 seconds per image
- **File Size Limit**: Recommended < 10MB
- **Supported Formats**: JPG, PNG, TIFF
- **Concurrent Requests**: Up to 4 simultaneous processing

---

**🎉 Your API is ready! Your friend can use these exact examples in Streamlit!**
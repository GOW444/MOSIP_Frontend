import streamlit as st
import requests
import json
import base64
from PIL import Image
import io

st.set_page_config(
    page_title="OCR API Documentation",
    page_icon=":material/api:",
    layout="wide"
)

st.title(":material/api: OCR Text Extraction API")
st.markdown("Complete API documentation and testing interface for the OCR service")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/description: Overview", 
    ":material/code: Endpoints", 
    ":material/play_arrow: Test API", 
    ":material/integration_instructions: Examples"
])

with tab1:
    st.header("API Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        The OCR Text Extraction API provides powerful text recognition capabilities using EasyOCR. 
        Upload images and receive structured text extraction results with confidence scores and bounding box coordinates.
        
        ### Key Features:
        - **Multi-format Support**: PNG, JPG, JPEG images
        - **Confidence Filtering**: Adjustable confidence thresholds
        - **Detailed Results**: Text regions with bounding boxes
        - **Statistics**: Comprehensive extraction metrics
        - **RESTful Design**: Standard HTTP methods and status codes
        """)
    
    with col2:
        st.info("""
        **API Base URL:**  
        `http://localhost:8000`
        
        **Content-Type:**  
        `multipart/form-data`
        
        **Response Format:**  
        `application/json`
        """)
    
    st.subheader("Quick Start")
    st.code("""
# Install dependencies
pip install fastapi uvicorn easyocr pillow opencv-python-headless

# Start API server
uvicorn api.ocr_api:app --reload --host 0.0.0.0 --port 8000

# Test endpoint
curl -X POST "http://localhost:8000/api/v1/extract-text" \\
     -F "file=@your_image.jpg" \\
     -F "min_confidence=0.1"
    """, language="bash")

with tab2:
    st.header("API Endpoints")
    
    # Health Check Endpoint
    with st.expander(":material/health_and_safety: GET /health", expanded=True):
        st.markdown("**Description:** Check API health and service information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Request:**")
            st.code("GET http://localhost:8000/health", language="http")
            
        with col2:
            st.markdown("**Response (200):**")
            st.code("""{
  "status": "healthy",
  "service": "OCR Text Extraction API",
  "supported_languages": ["en"],
  "supported_formats": ["PNG", "JPG", "JPEG"],
  "version": "1.0.0"
}""", language="json")
    
    # Main OCR Endpoint
    with st.expander(":material/image_search: POST /api/v1/extract-text", expanded=True):
        st.markdown("**Description:** Extract text from uploaded image")
        
        st.markdown("**Parameters:**")
        param_df = st.dataframe({
            "Parameter": ["file", "min_confidence"],
            "Type": ["File", "Float"],
            "Required": ["Yes", "No"],
            "Description": [
                "Image file (PNG, JPG, JPEG)",
                "Minimum confidence threshold (0.0-1.0, default: 0.1)"
            ]
        }, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Request:**")
            st.code("""POST /api/v1/extract-text
Content-Type: multipart/form-data

file: [image_file]
min_confidence: 0.1""", language="http")
        
        with col2:
            st.markdown("**Response (200):**")
            st.code("""{
  "status": "success",
  "message": "Text extraction completed successfully",
  "data": {
    "extracted_text": "Sample extracted text",
    "regions": [
      {
        "region_id": 1,
        "text": "Sample text",
        "confidence": 0.95,
        "bounding_box": {
          "top_left": [10, 20],
          "top_right": [100, 20],
          "bottom_right": [100, 40],
          "bottom_left": [10, 40]
        }
      }
    ],
    "statistics": {
      "total_regions": 1,
      "average_confidence": 0.95,
      "total_characters": 11,
      "min_confidence_used": 0.1
    },
    "metadata": {
      "filename": "image.jpg",
      "file_size_bytes": 15420,
      "image_dimensions": {
        "width": 800,
        "height": 600
      }
    }
  }
}""", language="json")

with tab3:
    st.header("Test API")
    st.markdown("Upload an image to test the OCR API directly from this interface")
    
    # API URL configuration
    api_url = st.text_input(
        "API Base URL", 
        value="http://localhost:8000",
        help="Make sure your API server is running on this URL"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Test Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to test OCR extraction"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Filter results below this confidence threshold"
        )
        
        if uploaded_file and st.button("Extract Text", icon=":material/image_search:", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Prepare API request
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {'min_confidence': min_confidence}
                    
                    # Make API request
                    response = requests.post(
                        f"{api_url}/api/v1/extract-text",
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        with col2:
                            st.subheader("API Response")
                            st.success(f"✅ {result['message']}")
                            
                            # Display extracted text
                            if result['data']['extracted_text']:
                                st.text_area(
                                    "Extracted Text",
                                    value=result['data']['extracted_text'],
                                    height=100
                                )
                                
                                # Display statistics
                                stats = result['data']['statistics']
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                
                                with stat_col1:
                                    st.metric("Regions", stats['total_regions'])
                                with stat_col2:
                                    st.metric("Avg Confidence", f"{stats['average_confidence']:.3f}")
                                with stat_col3:
                                    st.metric("Characters", stats['total_characters'])
                                
                                # Show detailed results
                                with st.expander("Detailed Results"):
                                    st.json(result)
                            else:
                                st.warning("No text detected with current confidence threshold")
                    
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API server. Make sure it's running!")
                except Exception as e:
                    st.error(f"❌ Request failed: {str(e)}")

with tab4:
    st.header("Usage Examples")
    
    # Python example
    with st.expander("Python (requests)", expanded=True):
        st.code("""
import requests

def extract_text_from_image(image_path, api_url="http://localhost:8000", min_confidence=0.1):
    \"\"\"Extract text from image using OCR API\"\"\"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'min_confidence': min_confidence}
        
        response = requests.post(
            f"{api_url}/api/v1/extract-text",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        return result['data']['extracted_text']
    else:
        raise Exception(f"API Error: {response.text}")

# Usage
try:
    text = extract_text_from_image("sample_image.jpg")
    print("Extracted text:", text)
except Exception as e:
    print("Error:", e)
        """, language="python")
    
    # cURL example
    with st.expander("cURL Command"):
        st.code("""
# Basic OCR extraction
curl -X POST "http://localhost:8000/api/v1/extract-text" \\
     -F "file=@sample_image.jpg" \\
     -F "min_confidence=0.1"

# Health check
curl -X GET "http://localhost:8000/health"

# Get supported languages
curl -X GET "http://localhost:8000/api/v1/supported-languages"
        """, language="bash")
    
    # JavaScript example
    with st.expander("JavaScript (Fetch API)"):
        st.code("""
async function extractTextFromImage(file, minConfidence = 0.1) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('min_confidence', minConfidence);
    
    try {
        const response = await fetch('http://localhost:8000/api/v1/extract-text', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            return result.data.extracted_text;
        } else {
            throw new Error(`API Error: ${response.status}`);
        }
    } catch (error) {
        console.error('OCR extraction failed:', error);
        throw error;
    }
}

// Usage with file input
document.getElementById('fileInput').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        try {
            const text = await extractTextFromImage(file);
            console.log('Extracted text:', text);
        } catch (error) {
            console.error('Error:', error);
        }
    }
});
        """, language="javascript")

# Instructions for running the API
st.header("Getting Started")

col1, col2 = st.columns(2)

with col1:
    st.subheader(":material/play_arrow: Start API Server")
    st.code("""
# Navigate to your project directory
cd your_project

# Install dependencies  
pip install fastapi uvicorn easyocr pillow opencv-python-headless

# Start the API server
uvicorn api.ocr_api:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# http://localhost:8000
    """, language="bash")

with col2:
    st.subheader(":material/check: Verify API")
    st.code("""
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "service": "OCR Text Extraction API",
  "version": "1.0.0"
}
    """, language="bash")

# API Status indicator
st.divider()
try:
    health_response = requests.get("http://localhost:8000/health", timeout=2)
    if health_response.status_code == 200:
        st.success("🟢 API Server is running and healthy!")
    else:
        st.warning("🟡 API Server responded but may have issues")
except:
    st.error("🔴 API Server is not running. Please start it using the instructions above.")

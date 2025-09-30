import streamlit as st
import requests
import json

st.set_page_config(
    page_title="API Documentation",
    page_icon="📚",
    layout="wide"
)

st.title("📚 API Documentation")
st.markdown("Complete guide to using the OCR Text Extraction API")
st.page_link("app.py", label="Back", icon=":material/chevron_backward:")
# API Base URL section
st.header("API Base URL")
api_url = st.text_input(
    "Enter API Base URL",
    value="http://localhost:8000",
    help="If you've deployed the API separately, enter its URL here. Default is localhost for testing."
)

st.info("💡 **Note**: The API server needs to be deployed separately from this Streamlit app. See deployment instructions below.")

st.divider()

# Overview
st.header("Overview")
st.markdown("""
This API provides two main endpoints for OCR text extraction and verification:

1. **Extract Text** (`/api/extract`) - Extract text from images
2. **Verify Text** (`/api/verify`) - Extract and verify text against expected output

Both endpoints support:
- Printed text recognition (EasyOCR)
- Handwritten text recognition (TrOCR)
- Confidence thresholding
- Detailed bounding box information
""")

st.divider()

# Endpoint 1: Extract Text
st.header("1. Extract Text")
st.subheader("`POST /api/extract`")

st.markdown("### Description")
st.markdown("Extracts text from an uploaded image and returns detected text regions with confidence scores.")

st.markdown("### Parameters")

params_df = {
    "Parameter": ["file", "text_type", "min_confidence"],
    "Type": ["File", "String", "Float"],
    "Required": ["✅ Yes", "❌ No", "❌ No"],
    "Default": ["-", "printed", "0.1"],
    "Description": [
        "Image file (PNG, JPG, JPEG)",
        "Type of text: 'printed' or 'handwritten'",
        "Minimum confidence threshold (0.0 to 1.0)"
    ]
}

st.table(params_df)

st.markdown("### Response")
st.code("""
{
  "success": true,
  "extracted_text": "Hello World",
  "total_regions": 2,
  "regions": [
    {
      "text": "Hello",
      "confidence": 0.95,
      "bounding_box": {
        "top_left": [10, 20],
        "top_right": [100, 20],
        "bottom_right": [100, 50],
        "bottom_left": [10, 50]
      }
    }
  ],
  "processing_time": 1.23,
  "text_type": "printed"
}
""", language="json")

st.markdown("### Try It Out")

with st.expander("🧪 Test Extract Endpoint", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        extract_file = st.file_uploader(
            "Upload Image for Extraction",
            type=['png', 'jpg', 'jpeg'],
            key="extract_upload"
        )
    
    with col2:
        extract_type = st.selectbox(
            "Text Type",
            ["printed", "handwritten"],
            key="extract_type"
        )
        extract_confidence = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.1,
            key="extract_confidence"
        )
    
    if st.button("Extract Text", key="extract_btn", type="primary"):
        if extract_file is None:
            st.error("Please upload an image first")
        else:
            try:
                with st.spinner("Processing..."):
                    files = {"file": extract_file.getvalue()}
                    data = {
                        "text_type": extract_type,
                        "min_confidence": extract_confidence
                    }
                    
                    response = requests.post(
                        f"{api_url}/api/extract",
                        files={"file": extract_file.getvalue()},
                        data=data,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Text extracted successfully!")
                        
                        st.subheader("Extracted Text")
                        st.text_area("Result", result["extracted_text"], height=100)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Regions", result["total_regions"])
                        with col_b:
                            st.metric("Processing Time", f"{result['processing_time']}s")
                        with col_c:
                            st.metric("Text Type", result["text_type"])
                        
                        st.subheader("Detailed Response")
                        st.json(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API server. Make sure it's running at: " + api_url)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("### Python Example")
st.code("""
import requests

# Prepare the image and parameters
with open('document.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'text_type': 'printed',
        'min_confidence': 0.1
    }
    
    response = requests.post(
        'http://your-api-url/api/extract',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Extracted Text: {result['extracted_text']}")
    print(f"Total Regions: {result['total_regions']}")
""", language="python")

st.markdown("### cURL Example")
st.code("""
curl -X POST "http://your-api-url/api/extract" \\
  -F "file=@document.jpg" \\
  -F "text_type=printed" \\
  -F "min_confidence=0.1"
""", language="bash")

st.divider()

# Endpoint 2: Verify Text
st.header("2. Verify Text")
st.subheader("`POST /api/verify`")

st.markdown("### Description")
st.markdown("Extracts text from an image and compares it against expected text, returning similarity metrics.")

st.markdown("### Parameters")

verify_params_df = {
    "Parameter": ["file", "expected_text", "text_type", "min_confidence"],
    "Type": ["File", "String", "String", "Float"],
    "Required": ["✅ Yes", "✅ Yes", "❌ No", "❌ No"],
    "Default": ["-", "-", "printed", "0.1"],
    "Description": [
        "Image file (PNG, JPG, JPEG)",
        "Expected text to compare against",
        "Type of text: 'printed' or 'handwritten'",
        "Minimum confidence threshold (0.0 to 1.0)"
    ]
}

st.table(verify_params_df)

st.markdown("### Response")
st.code("""
{
  "success": true,
  "extracted_text": "Hello World",
  "expected_text": "Hello World",
  "similarity_score": 100.0,
  "character_accuracy": 100.0,
  "matches": 11,
  "total_expected_chars": 11,
  "total_extracted_chars": 11,
  "verification_result": "excellent",
  "processing_time": 1.45,
  "text_type": "printed"
}
""", language="json")

st.markdown("### Verification Results")
st.markdown("""
- **excellent**: Similarity score ≥ 90%
- **good**: Similarity score ≥ 75%
- **fair**: Similarity score ≥ 50%
- **poor**: Similarity score < 50%
""")

st.markdown("### Try It Out")

with st.expander("🧪 Test Verify Endpoint", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        verify_file = st.file_uploader(
            "Upload Image for Verification",
            type=['png', 'jpg', 'jpeg'],
            key="verify_upload"
        )
        verify_expected = st.text_area(
            "Expected Text",
            placeholder="Enter the text you expect to see...",
            key="verify_expected"
        )
    
    with col2:
        verify_type = st.selectbox(
            "Text Type",
            ["printed", "handwritten"],
            key="verify_type"
        )
        verify_confidence = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.1,
            key="verify_confidence"
        )
    
    if st.button("Verify Text", key="verify_btn", type="primary"):
        if verify_file is None:
            st.error("Please upload an image first")
        elif not verify_expected.strip():
            st.error("Please enter expected text")
        else:
            try:
                with st.spinner("Processing and verifying..."):
                    data = {
                        "expected_text": verify_expected,
                        "text_type": verify_type,
                        "min_confidence": verify_confidence
                    }
                    
                    response = requests.post(
                        f"{api_url}/api/verify",
                        files={"file": verify_file.getvalue()},
                        data=data,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display verification result
                        if result["verification_result"] == "excellent":
                            st.success(f"✅ {result['verification_result'].upper()}: {result['similarity_score']}% match")
                        elif result["verification_result"] == "good":
                            st.info(f"ℹ️ {result['verification_result'].upper()}: {result['similarity_score']}% match")
                        else:
                            st.warning(f"⚠️ {result['verification_result'].upper()}: {result['similarity_score']}% match")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.subheader("Extracted Text")
                            st.text_area("Extracted", result["extracted_text"], height=100, key="result_extracted")
                        with col_b:
                            st.subheader("Expected Text")
                            st.text_area("Expected", result["expected_text"], height=100, key="result_expected")
                        
                        col_c, col_d, col_e = st.columns(3)
                        with col_c:
                            st.metric("Similarity Score", f"{result['similarity_score']}%")
                        with col_d:
                            st.metric("Character Accuracy", f"{result['character_accuracy']}%")
                        with col_e:
                            st.metric("Matches", f"{result['matches']}/{result['total_expected_chars']}")
                        
                        st.subheader("Detailed Response")
                        st.json(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API server. Make sure it's running at: " + api_url)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("### Python Example")
st.code("""
import requests

# Prepare the image and parameters
with open('document.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'expected_text': 'Hello World',
        'text_type': 'printed',
        'min_confidence': 0.1
    }
    
    response = requests.post(
        'http://your-api-url/api/verify',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Similarity Score: {result['similarity_score']}%")
    print(f"Verification Result: {result['verification_result']}")
    print(f"Extracted: {result['extracted_text']}")
""", language="python")

st.markdown("### cURL Example")
st.code("""
curl -X POST "http://your-api-url/api/verify" \\
  -F "file=@document.jpg" \\
  -F "expected_text=Hello World" \\
  -F "text_type=printed" \\
  -F "min_confidence=0.1"
""", language="bash")

st.divider()

# Health Check
st.header("3. Health Check")
st.subheader("`GET /api/health`")

st.markdown("### Description")
st.markdown("Check the health status of the API and verify which models are loaded.")

st.markdown("### Response")
st.code("""
{
  "status": "healthy",
  "easyocr_loaded": true,
  "trocr_loaded": true
}
""", language="json")

if st.button("Check API Health", key="health_btn"):
    try:
        response = requests.get(f"{api_url}/api/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ API is healthy!")
            st.json(result)
        else:
            st.error(f"Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to API server at: " + api_url)
        st.info("Make sure the API server is running and the URL is correct.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()

# Deployment Instructions
st.header("🚀 Deployment Instructions")

tab1, tab2, tab3 = st.tabs(["Local Development", "Deploy to Render", "Deploy to Railway"])

with tab1:
    st.markdown("""
    ### Local Development
    
    1. **Create `api_server.py`** with the FastAPI code
    
    2. **Create `requirements_api.txt`**:
    ```
    fastapi==0.104.0
    uvicorn[standard]==0.24.0
    python-multipart==0.0.6
    easyocr==1.7.0
    torch==2.0.0
    transformers==4.30.0
    pillow==10.0.0
    numpy==1.24.0
    opencv-python-headless==4.8.0
    ```
    
    3. **Install dependencies**:
    ```
    pip install -r requirements_api.txt
    ```
    
    4. **Run the API server**:
    ```
    python api_server.py
    ```
    
    5. The API will be available at `http://localhost:8000`
    
    6. **Test it**:
    ```
    curl http://localhost:8000/api/health
    ```
    """)

with tab2:
    st.markdown("""
    ### Deploy to Render (Free)
    
    1. **Sign up** at [render.com](https://render.com) (free with GitHub)
    
    2. **Push your code** to GitHub (make sure `api_server.py` and `requirements_api.txt` are in the repo)
    
    3. **Create New Web Service**:
       - Click "New +" → "Web Service"
       - Connect your GitHub repository
       - Fill in these settings:
    
    ```
    Name: mosip-ocr-api
    Environment: Python 3
    Region: Oregon (or closest to you)
    Branch: main
    
    Build Command:
    pip install -r requirements_api.txt
    
    Start Command:
    uvicorn api_server:app --host 0.0.0.0 --port $PORT
    ```
    
    4. **Select Free Plan**
    
    5. **Click "Create Web Service"**
    
    6. **Wait 5-10 minutes** for deployment
    
    7. **Get your URL**: `https://your-app-name.onrender.com`
    
    8. **Update the API URL** in the text box above with your Render URL
    
    ⚠️ **Note**: Free tier spins down after 15 minutes of inactivity. First request may take 30-60 seconds.
    """)

with tab3:
    st.markdown("""
    ### Deploy to Railway (Free)
    
    1. **Sign up** at [railway.app](https://railway.app) (free with GitHub)
    
    2. **Create New Project**:
       - Click "New Project"
       - Select "Deploy from GitHub repo"
       - Choose your repository
    
    3. Railway will **auto-detect** your Python app
    
    4. **Add environment variables** (if needed)
    
    5. **Click Deploy**
    
    6. **Get your URL** from the Settings → Domains section
    
    7. **Update the API URL** in the text box above with your Railway URL
    
    💡 **Tip**: Railway's free tier includes 500 hours/month and $5 credit.
    """)

st.divider()

# Error Codes
st.header("📋 Error Codes")

error_codes = {
    "Status Code": ["200", "400", "422", "500"],
    "Description": [
        "Success - Request processed successfully",
        "Bad Request - Invalid parameters provided",
        "Unprocessable Entity - Invalid file format or corrupted image",
        "Internal Server Error - Processing failed (check server logs)"
    ]
}

st.table(error_codes)

st.divider()

# Rate Limits
st.header("⚡ Rate Limits")
st.info("""
**Free Tier Limits**:
- Rate limits depend on your deployment platform
- Render Free: Limited by CPU/memory quotas
- Railway Free: 500 hours/month
- Local: No rate limits

For production use, consider upgrading to paid tiers.
""")

st.divider()

# Support
st.header("💬 Support & Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📖 Documentation**
    - [FastAPI Docs](https://fastapi.tiangolo.com)
    - [EasyOCR Docs](https://github.com/JaidedAI/EasyOCR)
    - [Transformers Docs](https://huggingface.co/docs/transformers)
    """)

with col2:
    st.markdown("""
    **🔧 Tools**
    - [API Testing with Postman](https://www.postman.com)
    - [cURL Documentation](https://curl.se/docs/)
    - [Requests Library](https://requests.readthedocs.io)
    """)

with col3:
    st.markdown("""
    **💡 Help**
    - [Streamlit Forum](https://discuss.streamlit.io)
    - [FastAPI Community](https://fastapi.tiangolo.com/community/)
    - GitHub Issues
    """)

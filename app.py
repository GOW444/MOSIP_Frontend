import streamlit as st
import requests
from PIL import Image

# This single file contains everything you need, fully updated to the new guides.

st.set_page_config(page_title="MOSIP OCR", page_icon="📄", layout="wide")

st.title("📄 MOSIP OCR - Text Extraction & Verification")
st.markdown("Upload document images to extract and validate text content using the live API.")

# --- API Configuration ---
# The new, live API URL provided by your friend [cite]
API_BASE = "https://deandra-creamiest-unpenetratingly.ngrok-free.dev"

# IMPORTANT: This header is required to bypass the ngrok browser warning page [cite]
HEADERS = {"ngrok-skip-browser-warning": "true"}

# --- Helper Function for API Connection Check ---
def check_api_status():
    try:
        response = requests.get(f"{API_BASE}/health", headers=HEADERS)
        if response.status_code == 200:
            st.sidebar.success("✅ API Connected")
            return True
        else:
            st.sidebar.error(f"API Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        st.sidebar.error("❌ API Connection Failed")
        return False

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")
check_api_status() # Check API status on load
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
preprocess = st.sidebar.checkbox("Image Preprocessing", value=True)

# --- Dynamic Language Selection ---
try:
    languages_response = requests.get(f"{API_BASE}/api/v1/languages", headers=HEADERS)
    if languages_response.status_code == 200:
        lang_data = languages_response.json()
        available_languages = list(lang_data['supported_languages'].keys())
        selected_languages = st.sidebar.multiselect("Languages", available_languages, default=["en"])
    else:
        st.sidebar.warning("Could not load languages. Using defaults.")
        selected_languages = ["en"]
except requests.exceptions.RequestException:
    st.sidebar.warning("API not ready. Using default languages.")
    selected_languages = ["en"]


# --- Main Interface ---
uploaded_file = st.file_uploader(
    "Upload Document Image",
    type=['jpg', 'jpeg', 'png', 'tiff'],
    help="Upload an image file containing text to extract"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        document_type = st.selectbox(
            "Document Type (for validation)",
            ["", "aadhaar", "pan", "passport", "driving_license", "voter_id"]
        )

    with col2:
        tab1, tab2, tab3 = st.tabs(["🔍 Extract Text", "✅ Validate Text", "🔄 Full Process"])

        # Tab 1: Simple Extraction
        with tab1:
            if st.button("Extract Text", key="extract", use_container_width=True):
                with st.spinner("Extracting text..."):
                    # Fixed file upload format for API compatibility
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "confidence_threshold": confidence_threshold,
                        "preprocess": preprocess,
                        "languages": ",".join(selected_languages)
                    }
                    try:
                        response = requests.post(f"{API_BASE}/api/v1/ocr/extract", files=files, data=data, headers=HEADERS)
                        if response.status_code == 200:
                            result = response.json()
                            st.metric("Text Blocks Found", result.get('text_blocks_found', 0))
                            st.text_area("📝 Extracted Text", result.get('combined_text', ''), height=150)
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection Error: Could not connect to the API at {API_BASE}.")

        # Tab 2: Manual Text Validation
        with tab2:
            text_input = st.text_area("Enter text to validate:", height=100)
            if st.button("Validate Text", key="validate", use_container_width=True) and text_input:
                with st.spinner("Validating..."):
                    payload = {"text": text_input, "document_type": document_type if document_type else None}
                    try:
                        response = requests.post(f"{API_BASE}/api/v1/ocr/validate", json=payload, headers=HEADERS)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(result.get('validation_message', 'Validation complete!'))
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection Error: Could not connect to the API at {API_BASE}.")

        # Tab 3: Combined Document Processing
        with tab3:
            validate_fields = st.checkbox("Validate extracted fields", value=True)
            if st.button("Process Complete Document", key="process", use_container_width=True, type="primary"):
                with st.spinner("Processing document..."):
                    # Fixed file upload format for API compatibility
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "confidence_threshold": confidence_threshold,
                        "document_type": document_type if document_type else None,
                        "validate_fields": validate_fields
                    }
                    try:
                        response = requests.post(f"{API_BASE}/api/v1/document/process", files=files, data=data, headers=HEADERS)
                        if response.status_code == 200:
                            result = response.json()
                            summary = result.get('summary', {})
                            st.success("Processing complete!")
                            st.text_area("📝 Validated Text", summary.get('combined_text', ''), height=120)
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection Error: Could not connect to the API at {API_BASE}.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("**MOSIP OCR API v1**")

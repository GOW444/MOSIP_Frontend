import streamlit as st
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import difflib

# Safe imports for optional features
TROCR_AVAILABLE = False
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    pass

# Set page configuration
st.set_page_config(
    page_title="OCR Text Extraction App",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for reader to avoid reloading
@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader and cache it"""
    return easyocr.Reader(['en'], gpu=False)  # Force CPU usage

@st.cache_resource
def load_trocr_models():
    """Load TrOCR models for handwritten text (CPU only)"""
    if not TROCR_AVAILABLE:
        return None, None
    
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()  # Set to evaluation mode
        return processor, model
    except Exception as e:
        st.error(f"Could not load TrOCR models: {e}")
        return None, None

def process_image_ocr(image_np, reader):
    """Process the image with EasyOCR and return results"""
    results = reader.readtext(image_np)
    return results

def process_handwritten_with_trocr(image_pil, reader, processor, model):
    """
    Use EasyOCR to detect text regions, then TrOCR to read handwritten text.
    Returns list of (bbox, text, confidence) tuples similar to EasyOCR format.
    """
    if processor is None or model is None:
        st.error("TrOCR models not available")
        return []
    
    try:
        # Convert PIL to numpy for EasyOCR
        image_np = np.array(image_pil)
        
        # Use EasyOCR to detect text regions (bounding boxes only)
        with st.spinner("Detecting text regions..."):
            easyocr_results = reader.readtext(image_np)
        
        if not easyocr_results:
            return []
        
        # Process each detected region with TrOCR
        trocr_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (bbox, _, confidence) in enumerate(easyocr_results):
            status_text.text(f"Processing region {idx + 1} of {len(easyocr_results)}...")
            
            # Extract bounding box coordinates
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image_pil.width, x_max + padding)
            y_max = min(image_pil.height, y_max + padding)
            
            # Crop the region
            cropped = image_pil.crop((x_min, y_min, x_max, y_max))
            
            # Process with TrOCR
            try:
                pixel_values = processor(cropped, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Store result in same format as EasyOCR
                trocr_results.append((bbox, text, confidence))
            except Exception as e:
                st.warning(f"Error processing region {idx + 1}: {e}")
                trocr_results.append((bbox, "[ERROR]", confidence))
            
            # Update progress
            progress_bar.progress((idx + 1) / len(easyocr_results))
        
        progress_bar.empty()
        status_text.empty()
        
        return trocr_results
        
    except Exception as e:
        st.error(f"Error in handwritten text processing: {e}")
        return []

def draw_bounding_boxes(image_np, results):
    """Draw bounding boxes and text on the image"""
    image_with_boxes = image_np.copy()
    
    for bbox, text, prob in results:
        # Extract the bounding box points
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        
        # Draw the rectangle
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
        
        # Put the detected text above the rectangle
        cv2.putText(image_with_boxes, text[:20], (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image_with_boxes

def calculate_character_similarity(extracted_text, expected_text):
    """Calculate character-by-character similarity and highlight differences"""
    extracted_clean = extracted_text.strip()
    expected_clean = expected_text.strip()
    
    total_chars_expected = len(expected_clean)
    total_chars_extracted = len(extracted_clean)
    
    matches = 0
    mismatches = []
    
    diff = list(difflib.ndiff(expected_clean, extracted_clean))
    
    expected_pos = 0
    extracted_pos = 0
    
    for item in diff:
        if item.startswith('  '):
            matches += 1
            expected_pos += 1
            extracted_pos += 1
        elif item.startswith('- '):
            mismatches.append({
                'type': 'missing',
                'position': expected_pos,
                'expected_char': item[2:],
                'actual_char': None
            })
            expected_pos += 1
        elif item.startswith('+ '):
            mismatches.append({
                'type': 'extra',
                'position': extracted_pos,
                'expected_char': None,
                'actual_char': item[2:]
            })
            extracted_pos += 1
    
    if total_chars_expected == 0:
        similarity_score = 1.0 if total_chars_extracted == 0 else 0.0
    else:
        similarity_score = matches / max(total_chars_expected, total_chars_extracted)
    
    return {
        'similarity_score': similarity_score,
        'matches': matches,
        'total_mismatches': len(mismatches),
        'mismatches': mismatches,
        'expected_length': total_chars_expected,
        'extracted_length': total_chars_extracted,
        'character_accuracy': (matches / total_chars_expected) * 100 if total_chars_expected > 0 else 0
    }

def highlight_text_differences(extracted_text, expected_text):
    """Create clear text comparison without colors, using only text formatting"""
    matcher = difflib.SequenceMatcher(None, expected_text, extracted_text)
    
    expected_html = ""
    extracted_html = ""
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        expected_chunk = expected_text[i1:i2]
        extracted_chunk = extracted_text[j1:j2]
        
        if tag == 'equal':
            expected_html += expected_chunk
            extracted_html += extracted_chunk
        elif tag == 'delete':
            expected_html += f"<del>{expected_chunk}</del>"
        elif tag == 'insert':
            extracted_html += f"<u>{extracted_chunk}</u>"
        elif tag == 'replace':
            expected_html += f"<del>{expected_chunk}</del>"
            extracted_html += f"<u>{extracted_chunk}</u>"
    
    return expected_html, extracted_html

def main():
    st.title("🔍 OCR Text Extraction App")
    
    # Check if API documentation page exists
    try:
        st.page_link("pages/02_API_Documentation.py", label="API Docs", icon=":material/docs:")
    except:
        pass
    
    st.markdown("Upload a PNG, JPG, or JPEG image to extract text using advanced OCR")
    
    # Main flow: Text type selection BEFORE file upload
    st.subheader("Step 1: Select Document Type")
    
    col_type1, col_type2 = st.columns(2)
    
    with col_type1:
        if TROCR_AVAILABLE:
            text_type = st.radio(
                "What type of text is in your document?",
                options=["Printed Text", "Handwritten Text"],
                help="Choose 'Handwritten Text' for better accuracy on handwritten documents",
                key="text_type_selector"
            )
        else:
            text_type = "Printed Text"
            st.radio(
                "What type of text is in your document?",
                options=["Printed Text"],
                help="Handwritten text recognition requires additional packages",
                disabled=True
            )
            st.info("ℹ️ Handwritten text recognition is not available. Using EasyOCR for printed text.")
    
    with col_type2:
        if text_type == "Handwritten Text":
            st.success("🖋️ **Handwritten Mode**")
            st.caption("Using EasyOCR for detection + TrOCR for recognition")
            st.caption("⚠️ Processing may take 1-3 minutes")
        else:
            st.success("🖨️ **Printed Text Mode**")
            st.caption("Using EasyOCR for fast and accurate text detection")
    
    st.divider()
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.1, 0.1)
        show_bounding_boxes = st.checkbox("Show bounding boxes", value=True)
    
    # File upload section
    st.subheader("Step 2: Upload Your Document")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image file"
    )
    
    if uploaded_file is not None:
        try:
            # Load the uploaded image
            image_pil = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image_pil)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_pil, caption="Uploaded Image", width='stretch')
            
            # Load OCR reader
            with st.spinner("Loading OCR models..."):
                reader = load_ocr_reader()
            
            # Process based on text type
            if text_type == "Handwritten Text" and TROCR_AVAILABLE:
                # Load TrOCR models
                processor, model = load_trocr_models()
                
                if processor is None or model is None:
                    st.error("Could not load handwritten text recognition models. Falling back to printed text mode.")
                    results = process_image_ocr(image_np, reader)
                else:
                    # Process with handwritten pipeline
                    st.info("🔄 Processing handwritten text... This may take a few minutes.")
                    results = process_handwritten_with_trocr(image_pil, reader, processor, model)
            else:
                # Standard EasyOCR processing for printed text
                with st.spinner("Processing image with OCR..."):
                    results = process_image_ocr(image_np, reader)
            
            # Filter results by confidence threshold
            filtered_results = [
                (bbox, text, prob) for bbox, text, prob in results 
                if prob >= min_confidence
            ]
            
            with col2:
                if show_bounding_boxes and filtered_results:
                    st.subheader("Detected Text Regions")
                    image_with_boxes = draw_bounding_boxes(image_np, filtered_results)
                    st.image(image_with_boxes, caption="Text Detection Results", width='stretch')
                else:
                    st.subheader("Processing Complete")
                    st.success(f"Found {len(filtered_results)} text regions")
            
            # Display results
            st.subheader("Extracted Text Results")
            
            if filtered_results:
                # Summary section
                all_text = " ".join([text for _, text, _ in filtered_results])
                st.text_area("Complete Extracted Text", value=all_text, height=100)
                
                # Text verification section
                st.divider()
                st.subheader(":material/verified: Text Verification")
                st.markdown("Enter the expected text to verify OCR accuracy")
                
                expected_text = st.text_area(
                    "Enter Expected Text",
                    placeholder="Paste or type the text you expect to see in the image...",
                    height=100,
                    help="Enter the complete text that should have been extracted from the image"
                )
                
                if expected_text.strip():
                    if st.button("Verify Text Accuracy", icon=":material/fact_check:", type="primary"):
                        with st.spinner("Analyzing text differences..."):
                            comparison_result = calculate_character_similarity(all_text, expected_text)
                            
                            col_score, col_metrics = st.columns([1, 1])
                            
                            with col_score:
                                score = comparison_result['similarity_score']
                                score_percent = score * 100
                                
                                if score_percent >= 90:
                                    st.success(f"✅ Excellent Match: {score_percent:.1f}%")
                                elif score_percent >= 75:
                                    st.warning(f"⚠️ Good Match: {score_percent:.1f}%")
                                elif score_percent >= 50:
                                    st.warning(f"⚠️ Fair Match: {score_percent:.1f}%")
                                else:
                                    st.error(f"❌ Poor Match: {score_percent:.1f}%")
                                
                                st.progress(score)
                            
                            with col_metrics:
                                st.metric("Character Accuracy", f"{comparison_result['character_accuracy']:.1f}%")
                                st.metric("Correct Characters", f"{comparison_result['matches']}")
                                st.metric("Total Mismatches", f"{comparison_result['total_mismatches']}")
                            
                            if comparison_result['total_mismatches'] > 0:
                                st.subheader("Detailed Text Comparison")
                                
                                expected_html, extracted_html = highlight_text_differences(all_text, expected_text)
                                
                                comp_col1, comp_col2 = st.columns(2)
                                
                                with comp_col1:
                                    st.markdown("**Expected Text:**")
                                    st.markdown(expected_html, unsafe_allow_html=True)
                                
                                with comp_col2:
                                    st.markdown("**Extracted Text:**")
                                    st.markdown(extracted_html, unsafe_allow_html=True)
                                
                                st.markdown("""
                                **Legend:**
                                - Normal text: Matching characters
                                - <del>Strikethrough</del>: Missing characters in extracted text
                                - <u>Underlined</u>: Extra/incorrect characters in extracted text
                                """, unsafe_allow_html=True)
                                
                                st.subheader("Error Analysis")
                                
                                if comparison_result['mismatches']:
                                    error_types = {'missing': 0, 'extra': 0}
                                    for mismatch in comparison_result['mismatches']:
                                        error_types[mismatch['type']] += 1
                                    
                                    error_col1, error_col2, error_col3 = st.columns(3)
                                    
                                    with error_col1:
                                        st.metric("Missing Characters", error_types['missing'])
                                    with error_col2:
                                        st.metric("Extra Characters", error_types['extra'])
                                    with error_col3:
                                        length_diff = abs(comparison_result['extracted_length'] - comparison_result['expected_length'])
                                        st.metric("Length Difference", length_diff)
                                
                                st.info("💡 **Common OCR Issues:** Characters like 'O' vs '0', 'I' vs 'l', 'rn' vs 'm' are frequently misread")
                            
                            else:
                                st.success("🎉 Perfect match! The extracted text exactly matches the expected text.")
                
                # Detailed results
                st.subheader("Detailed Detection Results")
                
                for i, (bbox, text, prob) in enumerate(filtered_results):
                    with st.expander(f"Region {i+1}: '{text[:30]}...' (Confidence: {prob:.2f})"):
                        col_text, col_details = st.columns([2, 1])
                        
                        with col_text:
                            st.write("**Detected Text:**")
                            st.code(text, language=None)
                        
                        with col_details:
                            if show_confidence:
                                st.metric("Confidence", f"{prob:.3f}")
                            st.write("**Bounding Box:**")
                            st.json({
                                "top_left": [int(bbox[0][0]), int(bbox[0][1])],
                                "top_right": [int(bbox[1][0]), int(bbox[1][1])],
                                "bottom_right": [int(bbox[2][0]), int(bbox[2][1])],
                                "bottom_left": [int(bbox[3][0]), int(bbox[3][1])]
                            })
                
                # Statistics
                st.subheader("Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Total Regions", len(filtered_results))
                
                with stats_col2:
                    avg_confidence = sum(prob for _, _, prob in filtered_results) / len(filtered_results)
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                
                with stats_col3:
                    total_chars = sum(len(text) for _, text, _ in filtered_results)
                    st.metric("Total Characters", total_chars)
                
                # Download option
                if st.button("Download Results as Text", icon=":material/download:"):
                    text_content = "\n".join([
                        f"Region {i+1} (Confidence: {prob:.3f}): {text}"
                        for i, (_, text, prob) in enumerate(filtered_results)
                    ])
                    st.download_button(
                        label="Download Text File",
                        data=text_content,
                        file_name=f"ocr_results_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain"
                    )
            
            else:
                st.warning("No text detected with the current confidence threshold. Try lowering the threshold or upload a clearer image.")
        
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.info("Please try uploading a different image or check if the image is valid.")
    
    else:
        st.info("👆 Upload an image file to get started!")
        
        with st.expander("How to use this app", icon=":material/info:"):
            st.markdown("""
            1. **Select Document Type**: Choose whether your image contains printed or handwritten text
            2. **Upload Image**: Click on 'Browse files' and select a PNG, JPG, or JPEG image
            3. **Adjust Settings**: Use the sidebar to modify confidence threshold and display options
            4. **View Results**: The app will automatically process your image and show:
               - Original image and detected text regions
               - Complete extracted text
               - Detailed results for each detected region
               - Statistics about the detection
            5. **Verify Accuracy**: Enter expected text to check OCR accuracy
            6. **Download**: Save the extracted text as a file
            
            **Tips for better results:**
            - Use high-resolution images with clear text
            - Ensure good contrast between text and background
            - Avoid blurry or distorted images
            - For handwritten text: Ensure neat handwriting and allow 1-3 minutes for processing
            - Printed text processes much faster than handwritten text
            """)

if __name__ == "__main__":
    main()

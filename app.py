import streamlit as st
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import difflib

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
    return easyocr.Reader(['en'])

def process_image_ocr(image_np, reader):
    """Process the image with OCR and return results"""
    results = reader.readtext(image_np)
    return results

def draw_bounding_boxes(image_np, results):
    """Draw bounding boxes and text on the image"""
    image_with_boxes = image_np.copy()
    
    for bbox, text, prob in results:
        # Extract the bounding box points
        top_left, top_right, bottom_right, bottom_left = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Draw the rectangle
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
        
        # Put the detected text above the rectangle
        cv2.putText(image_with_boxes, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return image_with_boxes

def calculate_character_similarity(extracted_text, expected_text):
    """Calculate character-by-character similarity and highlight differences"""
    # Normalize texts for comparison
    extracted_clean = extracted_text.strip()
    expected_clean = expected_text.strip()
    
    # Calculate basic metrics
    total_chars_expected = len(expected_clean)
    total_chars_extracted = len(extracted_clean)
    
    # Character-by-character comparison
    matches = 0
    mismatches = []
    
    # Use difflib for detailed comparison
    diff = list(difflib.ndiff(expected_clean, extracted_clean))
    
    # Process diff results
    expected_pos = 0
    extracted_pos = 0
    
    for item in diff:
        if item.startswith('  '):  # Match
            matches += 1
            expected_pos += 1
            extracted_pos += 1
        elif item.startswith('- '):  # Character in expected but missing in extracted
            mismatches.append({
                'type': 'missing',
                'position': expected_pos,
                'expected_char': item[2:],
                'actual_char': None
            })
            expected_pos += 1
        elif item.startswith('+ '):  # Extra character in extracted
            mismatches.append({
                'type': 'extra',
                'position': extracted_pos,
                'expected_char': None,
                'actual_char': item[2:]
            })
            extracted_pos += 1
    
    # Calculate similarity score
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
            # Matching text - plain text
            expected_html += expected_chunk
            extracted_html += extracted_chunk
        elif tag == 'delete':
            # Missing in extracted - strikethrough only
            expected_html += f"<del>{expected_chunk}</del>"
        elif tag == 'insert':
            # Extra in extracted - underline only
            extracted_html += f"<u>{extracted_chunk}</u>"
        elif tag == 'replace':
            # Different text - strikethrough for expected, underline for extracted
            expected_html += f"<del>{expected_chunk}</del>"
            extracted_html += f"<u>{extracted_chunk}</u>"
    
    return expected_html, extracted_html


def main():
    st.title("🔍 OCR Text Extraction App")
    st.page_link("pages/02_API_Documentation.py",label="API Docs",icon=":material/docs:")
    st.markdown("Upload a PNG, JPG, or JPEG image to extract text using EasyOCR")
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.1, 0.1)
        show_bounding_boxes = st.checkbox("Show bounding boxes", value=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image file"
    )
    
    if uploaded_file is not None:
        try:
            # Load the OCR reader
            with st.spinner("Loading OCR model..."):
                reader = load_ocr_reader()
            
            # Process the uploaded image
            image_pil = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image_pil)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_pil, caption="Uploaded Image", width="stretch")
            
            # Run OCR
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
                    st.image(image_with_boxes, caption="Text Detection Results", width="stretch")
                else:
                    st.subheader("Processing Complete")
                    st.success(f"Found {len(filtered_results)} text regions")
            
            # Display results
            st.subheader("Extracted Text Results")
            
            if filtered_results:
                # Summary section
                all_text = " ".join([text for _, text, _ in filtered_results])
                st.text_area("Complete Extracted Text", value=all_text, height=100)
                
                # NEW VERIFICATION SECTION
                st.divider()
                st.subheader(":material/verified: Text Verification")
                st.markdown("Enter the expected text to verify OCR accuracy")
                
                # User input for expected text
                expected_text = st.text_area(
                    "Enter Expected Text",
                    placeholder="Paste or type the text you expect to see in the image...",
                    height=100,
                    help="Enter the complete text that should have been extracted from the image"
                )
                
                if expected_text.strip():
                    if st.button("Verify Text Accuracy", icon=":material/fact_check:", type="primary"):
                        with st.spinner("Analyzing text differences..."):
                            # Calculate similarity
                            comparison_result = calculate_character_similarity(all_text, expected_text)
                            
                            # Display results
                            col_score, col_metrics = st.columns([1, 1])
                            
                            with col_score:
                                # Overall similarity score
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
                                
                                # Progress bar
                                st.progress(score)
                            
                            with col_metrics:
                                # Detailed metrics
                                st.metric("Character Accuracy", f"{comparison_result['character_accuracy']:.1f}%")
                                st.metric("Correct Characters", f"{comparison_result['matches']}")
                                st.metric("Total Mismatches", f"{comparison_result['total_mismatches']}")
                            
                            # Detailed comparison
                            if comparison_result['total_mismatches'] > 0:
                                st.subheader("Detailed Text Comparison")
                                
                                # Highlight differences
                                expected_html, extracted_html = highlight_text_differences(all_text, expected_text)
                                
                                comp_col1, comp_col2 = st.columns(2)
                                
                                with comp_col1:
                                    st.markdown("**Expected Text:**")
                                    st.markdown(expected_html, unsafe_allow_html=True)
                                
                                with comp_col2:
                                    st.markdown("**Extracted Text:**")
                                    st.markdown(extracted_html, unsafe_allow_html=True)
                                
                                # Legend
                                st.markdown("""
                                **Legend:**
                                - Normal text: Matching characters
                                - <del>Strikethrough</del>: Missing characters in extracted text
                                - <u>Underlined</u>: Extra/incorrect characters in extracted text
                                """, unsafe_allow_html=True)
                                
                                # Error analysis
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
                                
                                # Common OCR errors
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
                if st.button("Download Results as Text",icon=":material/download:"):
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
        # Instructions when no file is uploaded
        st.info("👆 Upload an image file to get started!")
        
        with st.expander("How to use this app",icon=":material/info:"):
            st.markdown("""
            1. **Upload Image**: Click on 'Browse files' and select a PNG, JPG, or JPEG image
            2. **Adjust Settings**: Use the sidebar to modify confidence threshold and display options
            3. **View Results**: The app will automatically process your image and show:
               - Original image and detected text regions
               - Complete extracted text
               - Detailed results for each detected region
               - Statistics about the detection
            4. **Verify Accuracy**: Enter expected text to check OCR accuracy
            5. **Download**: Save the extracted text as a file
            
            **Tips for better results:**
            - Use high-resolution images with clear text
            - Ensure good contrast between text and background
            - Avoid blurry or distorted images
            - Upload zoomed images if text is small
            """)

if __name__ == "__main__":
    main()



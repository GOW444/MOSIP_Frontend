import streamlit as st
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import difflib
import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from happytransformer import HappyTextToText, TTSettings

# Set page configuration
st.set_page_config(
    page_title="OCR Text Extraction App",
    page_icon="🔍",
    layout="wide"
)

# Avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize session state for readers to avoid reloading
@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader and cache it"""
    return easyocr.Reader(['en'])

@st.cache_resource
def load_trocr_models():
    """Load TrOCR processor and model and cache them"""
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    model.to(device)
    return processor, model

@st.cache_resource
def load_correction_model():
    """Load text correction model and cache it"""
    try:
        return HappyTextToText("T5", "google/flan-t5-large")
    except Exception as e:
        st.error(f"Failed to load correction model: {e}")
        return None

# TrOCR processing functions
def split_image_by_lines(image_pil, min_line_length_ratio=0.8, y_merge_threshold=10):
    """Detect long horizontal lines and split the image into horizontal strips."""
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        min_line_length = int(width * min_line_length_ratio)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_length, maxLineGap=20)

        # If no long lines found, return the full image as a single strip
        if lines is None:
            return [image_pil]

        # Collect unique Y coordinates
        y_coords = sorted(list(set([int(l[0][1]) for l in lines] + [int(l[0][3]) for l in lines])))

        # Merge close Y coordinates
        merged_y = []
        if y_coords:
            current_y = y_coords[0]
            for y in y_coords[1:]:
                if y - current_y < y_merge_threshold:
                    continue
                else:
                    merged_y.append(current_y)
                    current_y = y
            merged_y.append(current_y)

        image_strips = []
        last_y = 0
        split_points = merged_y + [height]

        for y in split_points:
            box = (0, last_y, width, y)
            cropped_image = image_pil.crop(box)
            image_strips.append(cropped_image)
            last_y = y

        return image_strips

    except Exception as e:
        st.error(f"Error in line cutting: {e}")
        return [image_pil]

def process_strips_with_trocr(image_strips, processor, model):
    """Runs TrOCR on a list of PIL.Image strips."""
    if not image_strips:
        return []

    extracted_text = []
    model.eval()

    for i, strip in enumerate(image_strips):
        try:
            inputs = processor(images=strip, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            extracted_text.append(text)
        except Exception as e:
            st.error(f"Could not process strip {i+1}: {e}")
            extracted_text.append("[ERROR]")

    return extracted_text

def extract_text_with_easyocr_trocr(image_pil, processor, model, y_threshold=15):
    """Uses EasyOCR to get text boxes, groups by lines, and runs TrOCR on grouped crops."""
    try:
        reader = easyocr.Reader(["en"], gpu=(device == "cuda"))
        
        # Convert PIL to numpy for EasyOCR
        image_np = np.array(image_pil)
        results = reader.readtext(image_np)

        if not results:
            return []

        # Group boxes by Y coordinate (approximate lines)
        lines = []
        for res in results:
            bbox, text, conf = res
            y_center = (bbox[0][1] + bbox[2][1]) / 2

            best_line_idx = None
            min_dist = float("inf")
            for i, line in enumerate(lines):
                dist = abs(y_center - line["y_center_avg"])
                if dist < y_threshold and dist < min_dist:
                    min_dist, best_line_idx = dist, i

            if best_line_idx is not None:
                lines[best_line_idx]["boxes"].append((bbox, text, conf))
                total_y = sum([(b[0][1] + b[2][1]) / 2 for b, _, _ in lines[best_line_idx]["boxes"]])
                lines[best_line_idx]["y_center_avg"] = total_y / len(lines[best_line_idx]["boxes"])
            else:
                lines.append({"boxes": [(bbox, text, conf)], "y_center_avg": y_center})

        # Sort lines top-to-bottom
        lines.sort(key=lambda x: x["y_center_avg"])

        extracted_text = []

        # Process each line with TrOCR
        for i, line in enumerate(lines):
            # Sort boxes left-to-right
            line["boxes"].sort(key=lambda x: x[0][0][0])

            # Compute bounding rect that contains all points of the line
            all_points = np.array([pt for (bbox, _, _) in line["boxes"] for pt in bbox])
            min_x, min_y = all_points.min(axis=0).astype(int)
            max_x, max_y = all_points.max(axis=0).astype(int)

            # Add padding and clamp to image bounds
            padding = 5
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image_pil.width, max_x + padding)
            max_y = min(image_pil.height, max_y + padding)

            # Crop and send to TrOCR
            cropped_image = image_pil.crop((min_x, min_y, max_x, max_y))

            try:
                inputs = processor(images=cropped_image, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                extracted_text.append(text)
            except Exception as e:
                st.error(f"Could not process line {i+1} with TrOCR: {e}")
                extracted_text.append("[ERROR]")

        return extracted_text

    except Exception as e:
        st.error(f"Error in EasyOCR + TrOCR method: {e}")
        return []

def correct_text_with_flan_t5(text_list, correction_model):
    """Uses FLAN-T5 for correcting OCR text."""
    if not text_list or not correction_model:
        return text_list

    settings = TTSettings(num_beams=5, min_length=1, max_length=200)
    corrected_text = []
    
    for text in text_list:
        prompt = f"""
Correct the spelling and character recognition errors in the following OCR output from a form field.

Example 1:
Input: 'Brenden : Female'
Output: 'Gender : Female'

Example 2:
Input: 'pincade : 560068 .'
Output: 'Pin Code : 560068'

Example 3:
Input: 'Email Id : AbigailO grnail . com'
Output: 'Email Id : Abigail@gmail.com'

Now correct this input exactly (preserve field labels where possible):

Input: '{text}'
Output:
"""
        try:
            result = correction_model.generate_text(prompt, args=settings)
            clean_result = result.text.strip()
        except Exception as e:
            st.error(f"Correction failed for '{text}': {e}")
            clean_result = text

        corrected_text.append(clean_result)

    return corrected_text

# Original EasyOCR functions (unchanged)
def process_image_ocr(image_np, reader):
    """Process the image with OCR and return results"""
    results = reader.readtext(image_np)
    return results

def draw_bounding_boxes(image_np, results):
    """Draw bounding boxes and text on the image"""
    image_with_boxes = image_np.copy()
    
    for bbox, text, prob in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image_with_boxes, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
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
    st.page_link("pages/02_API_Documentation.py", label="API Docs", icon=":material/docs:")
    st.markdown("Upload a PNG, JPG, or JPEG image to extract text using EasyOCR or TrOCR for handwritten text")
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("Settings")
        
        # NEW: Handwritten text option
        is_handwritten = st.checkbox("Handwritten Text", value=False, 
                                    help="Check this if your image contains handwritten text. This will use TrOCR which is better for handwritten content.")
        
        if is_handwritten:
            extraction_method = st.selectbox(
                "TrOCR Extraction Method",
                ["easyocr", "lines"],
                help="EasyOCR: Uses EasyOCR for text detection then TrOCR for recognition. Lines: Splits image by detected lines then uses TrOCR."
            )
            
            if extraction_method == "easyocr":
                line_grouping_threshold = st.slider("Line Grouping Threshold", 5, 30, 15, 1,
                                                   help="Pixels threshold for grouping text boxes into lines")
            
            enable_correction = st.checkbox("Enable Text Correction", value=True,
                                          help="Use FLAN-T5 to correct OCR errors")
        else:
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
            # Process the uploaded image
            image_pil = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image_pil)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            # Process based on handwritten option
            if is_handwritten:
                st.info(f"🖋️ Processing handwritten text using TrOCR with {extraction_method} method")
                
                # Load TrOCR models
                with st.spinner("Loading TrOCR models (this may take a while)..."):
                    processor, model = load_trocr_models()
                
                # Load correction model if needed
                correction_model = None
                if enable_correction:
                    with st.spinner("Loading text correction model..."):
                        correction_model = load_correction_model()
                
                # Run TrOCR processing
                with st.spinner("Processing with TrOCR..."):
                    if extraction_method == "lines":
                        image_strips = split_image_by_lines(image_pil)
                        original_text = process_strips_with_trocr(image_strips, processor, model)
                    else:  # easyocr method
                        original_text = extract_text_with_easyocr_trocr(image_pil, processor, model, 
                                                                      y_threshold=line_grouping_threshold)
                
                # Apply text correction if enabled
                if enable_correction and correction_model and original_text:
                    with st.spinner("Correcting text..."):
                        corrected_text = correct_text_with_flan_t5(original_text, correction_model)
                    final_text = corrected_text
                    show_original = True
                else:
                    final_text = original_text
                    show_original = False
                
                with col2:
                    st.subheader("Processing Complete")
                    st.success(f"Processed {len(final_text)} text regions using TrOCR")
                
                # Display results
                st.subheader("Extracted Text Results")
                
                if final_text:
                    # Combined text
                    all_text = " ".join(final_text)
                    st.text_area("Complete Extracted Text", value=all_text, height=100)
                    
                    # Show original vs corrected if correction was applied
                    if show_original and enable_correction:
                        st.subheader("Text Correction Results")
                        
                        for i, (orig, corr) in enumerate(zip(original_text, final_text)):
                            if orig != corr:
                                st.write(f"**Line {i+1}:**")
                                col_orig, col_corr = st.columns(2)
                                with col_orig:
                                    st.text_area(f"Original", value=orig, height=50, key=f"orig_{i}")
                                with col_corr:
                                    st.text_area(f"Corrected", value=corr, height=50, key=f"corr_{i}")
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    for i, text in enumerate(final_text):
                        with st.expander(f"Region {i+1}: '{text[:50]}...'"):
                            st.code(text, language=None)
                    
                else:
                    st.warning("No text detected in the image.")
            
            else:
                # Original EasyOCR processing
                with st.spinner("Loading EasyOCR model..."):
                    reader = load_ocr_reader()
                
                with st.spinner("Processing image with EasyOCR..."):
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
                        st.image(image_with_boxes, caption="Text Detection Results", use_column_width=True)
                    else:
                        st.subheader("Processing Complete")
                        st.success(f"Found {len(filtered_results)} text regions")
                
                # Display results (rest of original EasyOCR code)
                st.subheader("Extracted Text Results")
                
                if filtered_results:
                    all_text = " ".join([text for _, text, _ in filtered_results])
                    st.text_area("Complete Extracted Text", value=all_text, height=100)
                    
                    # Rest of the original verification and detailed results code...
                    # [Include all the original verification section code here]
                    
                else:
                    st.warning("No text detected with the current confidence threshold.")
            
            # Common verification section for both methods
            if (is_handwritten and final_text) or (not is_handwritten and filtered_results):
                # Get all text for verification
                if is_handwritten:
                    all_text = " ".join(final_text)
                else:
                    all_text = " ".join([text for _, text, _ in filtered_results])
                
                # Verification section (same as original)
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
                            
                            # Show detailed comparison if there are mismatches
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
                                
                                st.info("💡 **Common OCR Issues:** Characters like 'O' vs '0', 'I' vs 'l', 'rn' vs 'm' are frequently misread")
                            
                            else:
                                st.success("🎉 Perfect match! The extracted text exactly matches the expected text.")
                
                # Download option
                if st.button("Download Results as Text", icon=":material/download:"):
                    if is_handwritten:
                        text_content = "\n".join([f"Region {i+1}: {text}" for i, text in enumerate(final_text)])
                    else:
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
        
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.info("Please try uploading a different image or check if the image is valid.")
    
    else:
        st.info("👆 Upload an image file to get started!")
        
        with st.expander("How to use this app", icon=":material/info:"):
            st.markdown("""
            1. **Upload Image**: Click on 'Browse files' and select a PNG, JPG, or JPEG image
            2. **Choose Processing Type**: 
               - Uncheck 'Handwritten Text' for printed text (uses EasyOCR)
               - Check 'Handwritten Text' for handwritten content (uses TrOCR)
            3. **Adjust Settings**: Use the sidebar to modify options based on your text type
            4. **View Results**: The app will process your image and show extracted text
            5. **Verify Accuracy**: Enter expected text to check OCR accuracy
            6. **Download**: Save the extracted text as a file
            
            **Tips for better results:**
            - Use high-resolution images with clear text
            - For handwritten text, ensure good pen contrast
            - Avoid blurry or distorted images
            - Upload images with proper orientation
            """)

if __name__ == "__main__":
    main()

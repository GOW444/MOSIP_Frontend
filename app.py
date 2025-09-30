import streamlit as st
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io

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

def main():
    st.title("🔍 OCR Text Extraction App")
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
        
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload Image**: Click on 'Browse files' and select a PNG, JPG, or JPEG image
            2. **Adjust Settings**: Use the sidebar to modify confidence threshold and display options
            3. **View Results**: The app will automatically process your image and show:
               - Original image and detected text regions
               - Complete extracted text
               - Detailed results for each detected region
               - Statistics about the detection
            4. **Download**: Save the extracted text as a file
            
            **Tips for better results:**
            - Use high-resolution images with clear text
            - Ensure good contrast between text and background
            - Avoid blurry or distorted images
            """)

if __name__ == "__main__":
    main()




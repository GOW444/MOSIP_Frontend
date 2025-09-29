# 🚀 MOSIP OCR API - Quick Setup for Streamlit

## 📞 **For Your Friend:**

### 🌐 **Public API URL (Ready to Use!)**
```
https://deandra-creamiest-unpenetratingly.ngrok-free.dev
```

### ⚡ **Quick Test:**
1. **Open in browser**: https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/docs
2. **If you see an ngrok warning**, click "Visit Site" to proceed
3. **You should see the FastAPI documentation page**

### 🔧 **Important for Streamlit Code:**
**Always add this header to your requests to skip ngrok browser warnings:**
```python
headers = {"ngrok-skip-browser-warning": "true"}

# Example:
response = requests.post(
    "https://deandra-creamiest-unpenetratingly.ngrok-free.dev/api/v1/ocr/extract",
    files={"file": uploaded_file},
    data=data,
    headers=headers  # ← This is important!
)
```

### 📝 **Main Endpoints:**
1. **Extract Text**: `POST /api/v1/ocr/extract` 
2. **Validate Text**: `POST /api/v1/ocr/validate`
3. **Full Processing**: `POST /api/v1/document/process`
4. **Health Check**: `GET /health`
5. **API Docs**: `GET /api/docs`

### 🏃‍♂️ **Quick Streamlit Starter:**
```python
import streamlit as st
import requests

st.title("📄 MOSIP OCR Test")

# Add the required header
headers = {"ngrok-skip-browser-warning": "true"}
API_BASE = "https://deandra-creamiest-unpenetratingly.ngrok-free.dev"

# Test connection
if st.button("Test API Connection"):
    try:
        response = requests.get(f"{API_BASE}/health", headers=headers)
        if response.status_code == 200:
            st.success("✅ API is connected!")
            st.json(response.json())
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")

# File upload for OCR
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file and st.button("Extract Text"):
    files = {"file": uploaded_file}
    data = {"confidence_threshold": 0.7}
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/ocr/extract",
            files=files,
            data=data,
            headers=headers  # Important!
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Found {result['text_blocks_found']} text blocks")
            st.text_area("Extracted Text:", result['combined_text'])
        else:
            st.error(f"❌ Error: {response.text}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
```

### 🎯 **Key Points:**
- ✅ **No installation needed** - API is already running
- ✅ **Worldwide access** - Works from anywhere
- ✅ **Full documentation** - Available at `/api/docs`
- ⚠️ **Use headers** - Always include `ngrok-skip-browser-warning: true`
- ⚠️ **HTTPS only** - Use `https://` (not `http://`)

### 🛡️ **Troubleshooting:**
- **403/Browser Warning**: Add the ngrok header
- **Connection Refused**: Check if the API is still running
- **Timeout**: Processing may take 3-5 seconds for large images
- **CORS Issues**: API already has CORS enabled

### 📚 **Full Documentation:**
See `STREAMLIT_API_GUIDE.md` for complete examples and all endpoints.

---
**🎉 Ready to build! The API is live and waiting for your Streamlit app!**
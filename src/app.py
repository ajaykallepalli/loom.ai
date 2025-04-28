import streamlit as st
from google.cloud import storage
import requests
import uuid
import os
from PIL import Image
import io

# --- Configuration ---
# Replace with your actual bucket names and the URL of your deployed FastAPI service
USER_UPLOADS_BUCKET = "user-uploads-style-transfer-lab"
STYLE_IMAGES_BUCKET = "style-images-style-transfer-lab" # Assuming predefined styles might live here
RESULT_BUCKET_NAME = "stylized-results-style-transfer-lab" # For downloading results

# URL of the deployed FastAPI inference service (replace with actual URL after deployment)
# For local testing, if FastAPI runs on port 8080: "http://localhost:8080"
# For Cloud Run: Get the service URL from `gcloud run services describe ...`
# FASTAPI_URL = os.environ.get("FASTAPI_SERVICE_URL", "http://localhost:8080") 
FASTAPI_URL = "https://nst-fastapi-service-965013218383.us-central1.run.app" # Deployed URL
STYLIZE_ENDPOINT = f"{FASTAPI_URL}/stylize/"

# --- Google Cloud Storage Client ---
# Ensure ADC are set up for Streamlit app (e.g., service account if run on Cloud Run)
storage_client = None
def get_storage_client():
    global storage_client
    if storage_client is None:
        try:
            storage_client = storage.Client()
        except Exception as e:
            st.error(f"Failed to initialize Google Cloud Storage client: {e}")
            st.error("Ensure Application Default Credentials (ADC) are configured.")
            return None
    return storage_client

# --- GCS Helper Functions ---
def save_to_gcs(uploaded_file, bucket_name, user_id="default_user") -> str | None:
    """Uploads a Streamlit UploadedFile to GCS and returns the GCS URI."""
    client = get_storage_client()
    if not client:
        return None
        
    if uploaded_file is not None:
        try:
            bucket = client.bucket(bucket_name)
            # Use a unique name incorporating user ID and original filename
            blob_name = f"{user_id}/raw/{uuid.uuid4()}_{uploaded_file.name}"
            blob = bucket.blob(blob_name)
            
            # Reset file pointer before reading
            uploaded_file.seek(0)
            blob.upload_from_file(uploaded_file, content_type=uploaded_file.type)
            
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            print(f"Uploaded {uploaded_file.name} to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            st.error(f"Failed to upload {uploaded_file.name} to GCS: {e}")
            return None
    return None

def download_image_from_gcs(gcs_uri: str) -> Image.Image | None:
    """Downloads an image from GCS URI and returns it as a PIL Image."""
    client = get_storage_client()
    if not client or not gcs_uri:
        return None
        
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in RGB format
        if image.mode != 'RGB':
             image = image.convert('RGB')
        return image
    except Exception as e:
        st.error(f"Failed to download image from {gcs_uri}: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Neural Style Transfer")

st.info(f"Backend API Endpoint: {FASTAPI_URL}")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Content Image")
    content_file = st.file_uploader("Upload your content image", type=["jpg", "jpeg", "png"])
    if content_file:
        st.image(content_file, caption="Uploaded Content Image", use_column_width=True)

with col2:
    st.header("Style Image")
    # Option 1: Upload style image
    style_file = st.file_uploader("Upload your style image", type=["jpg", "jpeg", "png"])
    # Option 2: Select predefined style (TODO: Implement loading from GCS style bucket)
    # predefined_styles = list_styles_in_gcs(STYLE_IMAGES_BUCKET)
    # selected_style = st.selectbox("Or select a style", predefined_styles)
    
    if style_file:
        st.image(style_file, caption="Uploaded Style Image", use_column_width=True)
    # elif selected_style:
        # Load and display selected style image
        # style_image = download_image_from_gcs(f"gs://{STYLE_IMAGES_BUCKET}/{selected_style}")
        # if style_image: st.image(style_image, caption=f"Selected Style: {selected_style}", use_column_width=True)

with col3:
    st.header("Stylized Result")
    if st.button("Stylize!"): 
        if content_file and style_file: # For now, require both uploads
            with st.spinner("Uploading images and running style transfer..."):
                # 1. Upload images to GCS
                # TODO: Get a real user ID if authentication is added
                user_id = "streamlit_user"
                content_gcs_uri = save_to_gcs(content_file, USER_UPLOADS_BUCKET, user_id)
                style_gcs_uri = save_to_gcs(style_file, STYLE_IMAGES_BUCKET, user_id) # Or use predefined style URI

                if content_gcs_uri and style_gcs_uri:
                    st.write(f"Content: {content_gcs_uri}")
                    st.write(f"Style: {style_gcs_uri}")

                    # 2. Call FastAPI endpoint
                    try:
                        payload = {
                            "content_image_uri": content_gcs_uri,
                            "style_image_uri": style_gcs_uri
                        }
                        response = requests.post(STYLIZE_ENDPOINT, json=payload, timeout=120) # Increased timeout for potentially long GPU tasks
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                        result_data = response.json()
                        result_gcs_uri = result_data.get("result_image_uri")
                        st.write(f"Result: {result_gcs_uri}")

                        # 3. Download and display result from GCS
                        if result_gcs_uri:
                            result_image = download_image_from_gcs(result_gcs_uri)
                            if result_image:
                                st.image(result_image, caption="Stylized Result", use_column_width=True)
                            else:
                                st.error("Failed to download the result image.")
                        else:
                            st.error("Stylization request succeeded, but no result URI was returned.")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error calling style transfer API: {e}")
                        # Check if response is available for more details
                        if hasattr(e, 'response') and e.response is not None:
                           try: 
                             error_detail = e.response.json().get("detail", e.response.text)
                             st.error(f"API Error Detail: {error_detail}")
                           except: # Handle cases where response is not JSON
                              st.error(f"API Response Content: {e.response.text}") 
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.error("Failed to upload one or both images to GCS.")
        else:
            st.warning("Please upload both a content and a style image.")

# Add instructions or descriptions
st.sidebar.markdown("## How to Use")
st.sidebar.markdown("1. Upload a **Content Image**.")
st.sidebar.markdown("2. Upload a **Style Image**.")
st.sidebar.markdown("3. Click **Stylize!**")
st.sidebar.markdown("4. Wait for the result to appear.")
st.sidebar.markdown("--- ")
st.sidebar.markdown("**Note:** This is an MVP. Ensure the FastAPI backend is running and accessible.") 
import streamlit as st
from google.cloud import storage
import requests
import uuid
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="🎨 Loom.AI Style Transfer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
# Replace with your actual bucket names and the URL of your deployed FastAPI service
USER_UPLOADS_BUCKET = "user-uploads-style-transfer-lab"  # Bucket for user uploads
STYLE_IMAGES_BUCKET = "style-images-style-transfer-lab"  # Assuming predefined styles might live here
RESULT_BUCKET_NAME = "stylized-results-style-transfer-lab"  # For downloading results

# URL of the deployed FastAPI inference service
FASTAPI_URL = "https://nst-fastapi-service-965013218383.us-central1.run.app"  # Deployed URL
# FASTAPI_URL = "http://localhost:8080"  # Use this for local testing
STYLIZE_ENDPOINT = f"{FASTAPI_URL}/stylize/"
FAST_STYLIZE_ENDPOINT = f"{FASTAPI_URL}/stylize-fast/"  # New endpoint


# --- GCS Client (Cached) ---
# Cache the client initialization for efficiency
@st.cache_resource
def get_storage_client():
    try:
        client = storage.Client()
        print("Initialized GCS Client")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Google Cloud Storage client: {e}")
        st.error("Ensure Application Default Credentials (ADC) are configured or Service Account has permissions.")
        return None


storage_client = get_storage_client()


# --- GCS Helper Functions ---
def save_to_gcs(uploaded_file, bucket_name, user_id="streamlit_user") -> str | None:
    """Uploads a Streamlit UploadedFile to GCS and returns the GCS URI."""
    if not storage_client or not uploaded_file:
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
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


# Cache image downloads to avoid re-downloading the same result
@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_image_from_gcs(_client, gcs_uri: str) -> Image.Image | None:
    """Downloads an image from GCS URI and returns it as a PIL Image."""
    if not _client or not gcs_uri:
        return None
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = _client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print(f"Downloaded image from {gcs_uri}")
        return image
    except Exception as e:
        st.error(f"Failed to download image from {gcs_uri}: {e}")
        return None


# --- UI Layout --- 
st.title("🎨 Loom.AI Neural Style Transfer")
st.markdown("Transform your images with the artistic styles of famous paintings.")
st.markdown("--- ")

# === Option Selection ===
transfer_mode = st.sidebar.selectbox(
    "Choose Transfer Mode",
    ("Neural Style Transfer (Slow, Custom Style)", "Fast Style Transfer (Pre-trained Style)")
)
st.sidebar.markdown("--- ")

# Initialize session state for results
if 'slow_result_image' not in st.session_state:
    st.session_state.slow_result_image = None
if 'fast_result_image' not in st.session_state:
    st.session_state.fast_result_image = None

# =======================================
# ===== Neural Style Transfer (Slow) ====
# =======================================
if transfer_mode == "Neural Style Transfer (Slow, Custom Style)":
    st.header("🖌️ Neural Style Transfer (Custom Style)")
    st.markdown("Upload a content image and a style image. The style will be transferred to the content.")

    # --- Sidebar for Inputs & Settings ---
    st.sidebar.header("🖼️ Upload Images")
    content_file = st.sidebar.file_uploader("1. Content Image", type=["jpg", "jpeg", "png"], key="slow_content")
    style_file = st.sidebar.file_uploader("2. Style Image", type=["jpg", "jpeg", "png"], key="slow_style")

    st.sidebar.header("⚙️ Style Transfer Settings")
    # Resolution Selection
    resolution_option = st.sidebar.radio(
        "Output Resolution",
        ('Low (512px - Faster)', 'High (800px - Slower)'),
        index=0,  # Default to Low
        help="High resolution takes significantly longer and uses the result of the low-res pass as a starting point.",
        key="slow_resolution"
    )
    resolution_key = 'high' if resolution_option.startswith('High') else 'low'

    # Iterations Sliders
    st.sidebar.markdown("**Optimization Iterations**")
    iterations = st.sidebar.slider(
        "Low-Res Iterations",
        min_value=50,
        max_value=1000,
        value=300,  # Default from backend
        step=50,
        help="More iterations generally improve quality but increase processing time.",
        key="slow_iterations"
    )

    iterations_hr_disabled = (resolution_key == 'low')
    iterations_hr = st.sidebar.slider(
        "High-Res Iterations (if High selected)",
        min_value=50,
        max_value=500,
        value=200,  # Default from backend
        step=50,
        disabled=iterations_hr_disabled,
        help="Only applicable when High resolution is selected.",
        key="slow_iterations_hr"
    )
    
    # Style Strength Slider
    st.sidebar.markdown("**Style Strength**")
    style_weight = st.sidebar.slider(
        "Style Weight",
        min_value=1,
        max_value=100,
        value=10,  # Default value
        step=1,
        help="Higher values make the result look more like the style image. Lower values preserve more of the content image.",
        key="style_weight"
    )
    # Calculate the actual style weight (exponential scale)
    style_weight_value = 10 ** style_weight
    content_weight_value = 1e5  # Fixed content weight
    
    # Display the style/content balance
    st.sidebar.caption(f"Style/Content Balance: {style_weight}%")

    # --- Main Area for Previews and Results ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Content Image Preview")
        if content_file:
            st.image(content_file, use_container_width=True)
        else:
            st.info("Upload a Content Image using the sidebar.")

    with col2:
        st.subheader("Style Image Preview")
        if style_file:
            st.image(style_file, use_container_width=True)
        else:
            st.info("Upload a Style Image using the sidebar.")

    st.markdown("--- ")
    st.subheader("Stylized Result")

    # Placeholder for the result image display
    result_placeholder = st.empty()
    if st.session_state.slow_result_image:
        result_placeholder.image(st.session_state.slow_result_image, caption="Stylized Result", use_container_width=True)
    else:
        result_placeholder.info("Upload content and style images and click 'Stylize!' in the sidebar.")

    # Stylize Button (only active if both images are uploaded)
    if st.sidebar.button("✨ Stylize!", disabled=(not content_file or not style_file), use_container_width=True, key="slow_button"):
        result_placeholder.info("Processing... this may take several minutes depending on settings.")
        st.session_state.slow_result_image = None  # Clear previous result

        with st.spinner(f"Processing with {resolution_option} output..."):
            # 1. Upload images to GCS (only if storage client is available)
            user_id = "streamlit_user"
            content_gcs_uri = None
            style_gcs_uri = None
            if storage_client:
                content_gcs_uri = save_to_gcs(content_file, USER_UPLOADS_BUCKET, user_id)
                style_gcs_uri = save_to_gcs(style_file, STYLE_IMAGES_BUCKET, user_id)
            
            if content_gcs_uri and style_gcs_uri:
                st.sidebar.success("Images uploaded to GCS.")

                # 2. Call FastAPI endpoint
                try:
                    payload = {
                        "content_image_uri": content_gcs_uri,
                        "style_image_uri": style_gcs_uri,
                        "resolution": resolution_key,
                        "iterations": iterations,
                        "iterations_hr": iterations_hr,
                        "style_weight": style_weight_value,
                        "content_weight": content_weight_value
                    }
                    print(f"Sending request to {STYLIZE_ENDPOINT} with payload: {payload}")
                    
                    # Increase timeout significantly for potentially long GPU tasks
                    response = requests.post(
                        STYLIZE_ENDPOINT, 
                        json=payload, 
                        timeout=900
                    )  # 15 minutes timeout
                    response.raise_for_status()

                    result_data = response.json()
                    result_gcs_uri = result_data.get("result_image_uri")
                    st.sidebar.success("Stylization complete!")

                    # 3. Download and display result from GCS
                    if result_gcs_uri:
                        # Pass storage_client explicitly to cached function
                        result_image = download_image_from_gcs(storage_client, result_gcs_uri)
                        if result_image:
                            st.session_state.slow_result_image = result_image  # Store in session state
                            result_placeholder.image(
                                result_image,
                                caption=f"Stylized Result ({resolution_option})",
                                use_container_width=True
                            )
                        else:
                            result_placeholder.error("Failed to download the result image from GCS.")
                    else:
                        result_placeholder.error("Stylization request succeeded, but no result URI was returned.")

                except requests.exceptions.Timeout:
                    st.error("Error: The request timed out after 15 minutes. The process might be taking too long.")
                    result_placeholder.error("Request Timed Out.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling style transfer API: {e}")
                    error_detail = "No additional details available."
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_detail = e.response.json().get("detail", e.response.text)
                        except Exception:
                            error_detail = e.response.text
                    st.error(f"API Error Detail: {error_detail}")
                    result_placeholder.error("API Request Failed.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    result_placeholder.error("An unexpected error occurred.")
            else:
                st.error("Failed to upload one or both images to GCS. Cannot proceed.")
                result_placeholder.warning("Image upload failed.")

# =======================================
# ===== Fast Style Transfer ============
# =======================================
else:  # Fast Transfer
    st.header("⚡ Fast Style Transfer (Pre-trained Style)")
    st.markdown("Apply a pre-trained style model to your content image. Much faster than custom style transfer.")

    # --- Sidebar for Inputs & Settings ---
    st.sidebar.header("🖼️ Upload Content Image")
    fast_content_file = st.sidebar.file_uploader("Content Image", type=["jpg", "jpeg", "png"], key="fast_content")

    st.sidebar.header("⚙️ Fast Style Settings")
    # Style Model Selection
    style_model = st.sidebar.selectbox(
        "Select Style",
        options=["starry_night", "kandinsky", "scream", "gauguin", "ed_hopper"],
        index=0,  # Default to starry_night
        format_func=lambda x: {
            "starry_night": "🌠 Starry Night (Van Gogh)",
            "kandinsky": "🎨 Composition (Kandinsky)",
            "scream": "😱 The Scream (Munch)",
            "gauguin": "🏝️ Tahitian (Gauguin)",
            "ed_hopper": "🏙️ Nighthawks (Hopper)"
        }.get(x, x),
        key="fast_style_model"
    )

    # Resolution Options
    use_full_size = st.sidebar.checkbox(
        "Use Original Resolution",
        value=False,
        help="When enabled, maintains original image resolution. May be slow for very large images.",
        key="fast_use_full_size"
    )

    resize_dim = st.sidebar.slider(
        "Resize Dimension",
        min_value=256,
        max_value=1024,
        value=512,
        step=64,
        disabled=use_full_size,
        help="Maximum width/height the image will be resized to while preserving aspect ratio.",
        key="fast_resize_dim"
    )

    # --- Main Area for Previews and Results ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Content Image Preview")
        if fast_content_file:
            st.image(fast_content_file, use_container_width=True)
        else:
            st.info("Upload a Content Image using the sidebar.")

    with col2:
        st.subheader("Style Preview")
        # Map style model to correct image filename
        style_image_map = {
            "starry_night": "starry_night.jpg",
            "kandinsky": "kandinsky_1.jpg",
            "scream": "scream_1.jpg",
            "gauguin": "gauguin_1.jpg",
            "ed_hopper": "Nighthawks_EdwardHopper.jpg"
        }
        style_image_path = f"images/{style_image_map.get(style_model, style_model + '.jpg')}"
        try:
            style_preview = Image.open(style_image_path)
            st.image(style_preview, caption=f"Style: {style_model}", use_container_width=True)
        except Exception:
            st.warning(f"Style preview image not found: {style_image_path}")

    st.markdown("--- ")
    st.subheader("Stylized Result")

    # Placeholder for result display
    fast_result_placeholder = st.empty()
    if st.session_state.fast_result_image:
        fast_result_placeholder.image(
            st.session_state.fast_result_image,
            caption="Stylized Result",
            use_container_width=True
        )
    else:
        fast_result_placeholder.info("Upload a content image and click 'Apply Style' in the sidebar.")

    # Stylize Button (only active if content image uploaded)
    if st.sidebar.button("✨ Apply Style", disabled=not fast_content_file, use_container_width=True, key="fast_button"):
        fast_result_placeholder.info("Processing... this should only take a few seconds.")
        st.session_state.fast_result_image = None  # Clear previous result

        with st.spinner(f"Applying {style_model} style..."):
            try:
                if not fast_content_file:
                    raise ValueError("No content image provided.")

                # Direct upload approach - prepare form data
                files = {'content_image': (fast_content_file.name, fast_content_file.getvalue(), fast_content_file.type)}
                
                data = {
                    'model_name': style_model,
                    'use_full_size': str(use_full_size),
                    'resize_dim': str(resize_dim)
                }
                
                # Call FastAPI endpoint - simpler direct approach with response as image
                response = requests.post(
                    FAST_STYLIZE_ENDPOINT,
                    files=files,
                    data=data,
                    timeout=60  # 60 second timeout should be plenty for fast method
                )
                response.raise_for_status()

                # Success - process the image bytes directly from response
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    st.session_state.fast_result_image = image  # Store in session state
                    fast_result_placeholder.image(image, caption=f"Style: {style_model}", use_container_width=True)
                    
                    # Display actual size that was processed
                    st.sidebar.success(f"Success! Image size: {image.width}x{image.height}")
                else:
                    st.error(f"Error: Received status code {response.status_code}")
                    fast_result_placeholder.error("Style transfer failed.")
                    
            except requests.exceptions.Timeout:
                st.error("Error: The request timed out. The server might be busy or the image too large.")
                fast_result_placeholder.error("Request Timed Out.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error calling fast style transfer API: {e}")
                fast_result_placeholder.error("API Request Failed.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                fast_result_placeholder.error("An unexpected error occurred.")

# --- Footer/Info (Common) ---
st.sidebar.markdown("--- ")
st.sidebar.caption(f"Backend: {FASTAPI_URL}")
st.sidebar.markdown("--- ")
st.sidebar.markdown("**Note:** Ensure the FastAPI backend is running and accessible.") 
import io
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from google.cloud import storage
import torch
from pydantic import BaseModel # Import BaseModel

# Use relative import because both files are in the 'src' directory/package
from .style_transfer import run_style_transfer # Correct function name

app = FastAPI()

# --- Pydantic Model for Request Body ---
class StylizeRequest(BaseModel):
    content_image_uri: str
    style_image_uri: str
    # Add other parameters like resolution if needed in the future
    # resolution: str | None = 'low' 

# --- Configuration ---
# TODO: Replace with your actual bucket names
# Make sure the Service Account running this has access
CONTENT_BUCKET_NAME = "user-uploads-style-transfer-lab" # Bucket for user-uploaded content images
STYLE_BUCKET_NAME = "style-images-style-transfer-lab"   # Bucket for style images (can be public)
RESULT_BUCKET_NAME = "stylized-results-style-transfer-lab" # Bucket to store results

# --- Model Loading ---
# The style_transfer module now loads the VGG model internally
# We just need to ensure the device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Google Cloud Storage Client ---
# Initialized lazily or globally if preferred
# Ensure Application Default Credentials (ADC) are set up where this runs
# (e.g., inherited from Cloud Run service account)
storage_client = None

def get_storage_client():
    global storage_client
    if storage_client is None:
        storage_client = storage.Client()
    return storage_client

# --- Helper Functions ---
def download_image_from_gcs(bucket_name: str, blob_name: str) -> Image.Image:
    """Downloads an image from GCS and returns it as a PIL Image."""
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in RGB format
        if image.mode != 'RGB':
             image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error downloading GCS image gs://{bucket_name}/{blob_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download image: {blob_name}")

def upload_image_to_gcs(image: Image.Image, bucket_name: str, blob_name: str) -> str:
    """Uploads a PIL image to GCS and returns the public URL (or GCS URI)."""
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG') # Or PNG, adjust as needed
        img_byte_arr.seek(0)

        blob.upload_from_file(img_byte_arr, content_type='image/jpeg')
        
        # Return GCS URI - frontend/caller might need this format
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        print(f"Uploaded result to {gcs_uri}")
        return gcs_uri
        # Alternatively, return a signed URL or public URL if bucket is public
        # return blob.public_url 
    except Exception as e:
        print(f"Error uploading GCS image gs://{bucket_name}/{blob_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload result image.")


@app.post("/stylize/")
async def stylize_endpoint(request: StylizeRequest):
    """
    Applies style transfer given GCS URIs for content and style images in the request body.
    Example URIs: gs://user-uploads-style-transfer-lab/user123/raw/cat.jpg, gs://style-images-style-transfer-lab/vangogh.jpg
    """
    print(f"Received request: content='{request.content_image_uri}', style='{request.style_image_uri}'")
    
    try:
        # Parse GCS URIs from the request model
        content_bucket, content_blob = request.content_image_uri.replace("gs://", "").split("/", 1)
        style_bucket, style_blob = request.style_image_uri.replace("gs://", "").split("/", 1)

        # Download images from GCS
        content_image = download_image_from_gcs(content_bucket, content_blob)
        style_image = download_image_from_gcs(style_bucket, style_blob)
        
        print(f"Content image size: {content_image.size}, Style image size: {style_image.size}")

        # Perform style transfer using the imported function
        # The run_style_transfer function handles model loading and device placement internally
        # It returns (low_res_image, high_res_image). We only use low-res for now.
        result_image, _ = run_style_transfer(content_img=content_image, 
                                             style_img=style_image, 
                                             resolution='low') # Explicitly request low resolution

        if result_image is None:
             raise HTTPException(status_code=500, detail="Style transfer failed to produce an image.")

        # Generate a unique name for the result
        result_filename = f"stylized_{uuid.uuid4()}.jpg"
        # Store in a structured path if desired, e.g., based on content blob path
        result_blob_name = f"results/{result_filename}" 

        # Upload result to GCS
        result_gcs_uri = upload_image_to_gcs(result_image, RESULT_BUCKET_NAME, result_blob_name)

        return JSONResponse(content={"result_image_uri": result_gcs_uri})

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        print(f"Error during stylization: {e}")
        # Log the full error traceback here in a real scenario
        import traceback; traceback.print_exc();
        raise HTTPException(status_code=500, detail=f"Internal server error during stylization: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# You might want to add model loading logic within a startup event
# @app.on_event("startup")
# async def startup_event():
#     global model, device
#     # model = load_my_model() # Load your pre-trained model here
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model.to(device).eval()
#     print("Model loaded on startup (placeholder).")

if __name__ == "__main__":
    # Need to adjust how this runs locally due to relative import
    # Typically run via `python -m src.inference` from the root directory
    print("To run locally, execute from the project root: python -m uvicorn src.inference:app --reload --host 0.0.0.0 --port 8080")
    # The Dockerfile CMD handles running it correctly in the container.
    pass 
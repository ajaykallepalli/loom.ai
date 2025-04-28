# Loom.AI

![Loom.AI Banner](assets/banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI/CD](https://github.com/username/loom.ai/actions/workflows/cicd.yml/badge.svg)](https://github.com/username/loom.ai/actions/workflows/cicd.yml)

> Loom.AI: Transform your images with the artistic styles of world-renowned masterpieces

## 🎨 Overview

Loom.AI is an advanced image styling platform that uses neural style transfer techniques to transform ordinary photos into artistic masterpieces. Upload any image and choose from a curated collection of artistic styles ranging from Van Gogh to Picasso.

![Demo GIF](assets/demo.gif)

## 🔥 Features

- **Neural Style Transfer**: Transform images using state-of-the-art style transfer models
- **Multiple Style Options**: Choose from various pre-trained artistic styles
- **Custom Style Upload**: Upload your own style images (premium feature)
- **Batch Processing**: Process multiple images at once
- **Adjustable Parameters**: Fine-tune styling intensity, resolution, and more
- **History Tracking**: View and download your previously styled images

## 🛠️ Architecture

```
+--------------------------------------------------------------+
|                         Data Sources                         |
|  - Style Images (Wikimedia Commons)                          |
|  - Content Images (MS COCO, User Uploads)                    |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|           Data Versioning & Orchestration Layer              |
|  - DVC: Track versions of data/models (GCS remote)           |
|  - Metaflow: Orchestrates pipeline steps                     |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|                  ML Training & Preprocessing                 |
|  - Environment: GCP Vertex AI Workbench / GCE VM (GPU)       |
|  - Framework: PyTorch (NST / AdaIN / Transformer)            |
|  - Preprocessing: Resize, normalize, batch                   |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|                 Experiment Tracking & Registry               |
|  - MLflow: Track metrics, loss, outputs                      |
|  - MLflow Model Registry (on GCS + Cloud SQL)                |
|  - Optional: Sync with Vertex AI Model Registry              |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|                        Model Serving                         |
|  - FastAPI: Serve `/stylize` API endpoint                    |
|  - Load model from GCS or MLflow Registry                    |
|  - Containerized with Docker                                 |
|  - Deployed on Cloud Run or GKE                              |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|                   API Layer & Frontend                       |
|  - FastAPI Backend: REST API for upload, styles, history     |
|  - Streamlit Frontend (Cloud Run / GCE):                     |
|     • Upload image                                           |
|     • Select art style(s)                                    |
|     • View/download output                                   |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|                Additional Services / Infrastructure          |
|  - Logging: Cloud Logging (GCP)                              |
|  - Monitoring: Cloud Monitoring / Prometheus (GKE)           |
|  - Data Encryption: GCS (at-rest, optional CMEK)             |
|  - CI/CD: GitHub Actions → Docker Build → Cloud Run Deploy   |
+--------------------------------------------------------------+
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Google Cloud Platform account (for production deployment)
- Git LFS (for model storage)

### Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/ynaung24/loom.ai.git
cd loom.ai
```

2. **Set up environment**

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

3. **Set up DVC**

```bash
# Initialize DVC
dvc init

# Add GCS remote storage (optional)
dvc remote add -d storage gs://loom-ai-data
```

4. **Run the application locally**

```bash
# Start the API server
python -m api.main

# In a separate terminal, start the Streamlit frontend
python -m frontend.app
```

5. **Access the application**

Open your browser and navigate to:
- Frontend UI: http://localhost:xxxx
- API Documentation: http://localhost:xxxx/docs

## ☁️ GCP Setup (for Cloud Run Deployment)

Deploying the FastAPI backend to Cloud Run with GPU support requires specific GCP resources and configuration. These steps mirror the first few days of the learning path outlined in the initial project brief.

**Prerequisites:**

*   Google Cloud SDK (`gcloud`) installed and authenticated.
*   A GCP project with **Billing Enabled**.

**Steps:**

1.  **Create/Set Project:**
    *   Choose a unique Project ID (e.g., `style-transfer-lab`).
    ```bash
    # Create project (if it doesn't exist)
    gcloud projects create YOUR_PROJECT_ID

    # Set project as default for gcloud commands
    gcloud config set project YOUR_PROJECT_ID
    ```

2.  **Enable APIs:** Enable necessary APIs for the project.
    ```bash
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com cloudbuild.googleapis.com logging.googleapis.com monitoring.googleapis.com
    ```

3.  **Create GCS Buckets:** Create buckets for uploads, styles, and results. 
    *   **Important:** Bucket names must be globally unique. The commands below append `-YOUR_PROJECT_ID` for uniqueness. 
    *   **Configuration:** You *must* update the hardcoded bucket names in `src/inference.py` and `src/app.py` to match the actual names you create.
    *   *Note:* Buckets are created in `us-west1`, but the GPU service runs in `us-central1`. Co-locating might be preferable in a production setup.
    ```bash
    # Bucket for user content uploads
    gcloud storage buckets create gs://user-uploads-YOUR_PROJECT_ID --project=YOUR_PROJECT_ID --location=us-west1

    # Bucket for style images
    gcloud storage buckets create gs://style-images-YOUR_PROJECT_ID --project=YOUR_PROJECT_ID --location=us-west1

    # Bucket for storing stylized results
    gcloud storage buckets create gs://stylized-results-YOUR_PROJECT_ID --project=YOUR_PROJECT_ID --location=us-west1

    # Make style images bucket publicly readable (optional, for sample styles)
    gcloud storage buckets add-iam-policy-binding gs://style-images-YOUR_PROJECT_ID --member=allUsers --role=roles/storage.objectViewer
    ```
    *(Remember to replace `YOUR_PROJECT_ID` with your actual project ID in the commands and update the code accordingly!)*

4.  **Create Artifact Registry Repository:** Create a Docker repository to store the built images.
    ```bash
    gcloud artifacts repositories create style-repo --repository-format=docker --location=us-west1 --description="Style transfer Docker repository"
    ```

5.  **GPU Quota:** Cloud Run GPU deployment requires specific quotas.
    *   The current setup requires **Nvidia L4 GPU** quota with **zonal redundancy disabled** in the `us-central1` region.
    *   Check your quota in the GCP Console under "IAM & Admin" -> "Quotas". Filter for "Cloud Run Admin API" service and look for relevant GPU limits in `us-central1`.
    *   If your limit is 0, you must request an increase (e.g., to 1 or more). Requests may take time to be approved.

**Using the Deployed Application (for Teammates):**

If the main FastAPI backend has already been deployed to Cloud Run (as described below), teammates can run the Streamlit user interface locally to interact with the application without needing to perform the full GCP setup or deploy the backend themselves.

1.  **Prerequisites:**
    *   Python 3.8+ installed.
    *   Git installed.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/ynaung24/loom.ai.git
    cd loom.ai
    ```
3.  **Set up Python environment & Install dependencies:**
    ```bash
    # Create and activate a virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate # Or venv\Scripts\activate on Windows

    # Install required packages
    pip install -r requirements.txt
    ```
4.  **Configure Frontend:**
    *   Open the file `src/app.py`.
    *   Find the `FASTAPI_URL` variable.
    *   Make sure it is set to the deployed backend URL:
        ```python
        FASTAPI_URL = "https://nst-fastapi-service-965013218383.us-central1.run.app"
        ```
5.  **Run Streamlit Frontend:**
    ```bash
    streamlit run src/app.py
    ```
    Your browser should open, and the Streamlit app will communicate with the deployed GPU backend on Cloud Run.

**Running Locally (with GCP Buckets):**

*   Ensure you have authenticated for Application Default Credentials (ADC):
    ```bash
    gcloud auth application-default login
    ```
*   Make sure the logged-in user has permissions on the created GCS buckets.
*   Run the FastAPI backend:
    ```bash
    # From project root
    python -m uvicorn src.inference:app --reload --host 0.0.0.0 --port 8080
    ```
*   Run the Streamlit frontend (in a separate terminal):
    ```bash
    # From project root
    streamlit run src/app.py
    ```
*   The Streamlit app will use the locally running backend but store/retrieve images from the configured GCS buckets.

**Deploying to Cloud Run (GPU):**

1.  **Build and Push Image:**
    ```bash
    gcloud builds submit --tag us-west1-docker.pkg.dev/YOUR_PROJECT_ID/style-repo/nst-fastapi:latest .
    ```
2.  **Configure `service.yaml`:** This file contains the necessary configuration to deploy with an L4 GPU and disable zonal redundancy.
    *   Ensure `metadata.name` is desired service name (e.g., `nst-fastapi-service`).
    *   Ensure `spec.template.spec.containers.image` points to the correct image URL built above.
    *   The file requires: 4 CPU, 16Gi Memory, `maxScale: '1'`, and specifies `nvidia-l4`.
3.  **Deploy using YAML:**
    ```bash
    gcloud run services replace service.yaml --region=us-central1
    ```
4.  **Allow Public Access (if needed):**
    ```bash
    gcloud run services add-iam-policy-binding nst-fastapi-service --member=allUsers --role=roles/run.invoker --region=us-central1
    ```
5.  **Update Frontend Config:** Update `FASTAPI_URL` in `src/app.py` to point to the deployed Cloud Run service URL (e.g., `https://nst-fastapi-service-....run.app`).

## 🧪 Training Models

If you want to train custom style transfer models:

```bash
# Pull the training data
dvc pull data/training

# Run the training pipeline
python -m ml.train --config configs/training/adain.yaml
```

## 📦 Project Structure

```
loom.ai/
├── api/                    # FastAPI backend
│   ├── main.py             # API entrypoint
│   ├── models/             # Pydantic models
│   ├── routes/             # API routes
│   └── services/           # Business logic
├── data/                   # Data directory (tracked by DVC)
│   ├── content/            # Content images for training/testing
│   ├── styles/             # Style images
│   └── processed/          # Processed data
├── frontend/               # Streamlit frontend
│   ├── app.py              # Main Streamlit application
│   ├── components/         # UI components
│   └── utils/              # Frontend utilities
├── ml/                     # Machine learning code
│   ├── models/             # PyTorch model definitions
│   ├── train.py            # Training script
│   ├── predict.py          # Inference script
│   └── preprocessing/      # Data preprocessing functions
├── flows/                  # Metaflow pipelines
│   ├── training_flow.py    # Model training flow
│   └── inference_flow.py   # Batch inference flow
├── notebooks/              # Jupyter notebooks for exploration
├── infra/                  # Infrastructure code
│   ├── docker/             # Dockerfiles
│   ├── k8s/                # Kubernetes manifests
│   └── terraform/          # Terraform IaC
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── assets/                 # Static assets
├── .dvc/                   # DVC configuration
├── .github/                # GitHub Actions workflows
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── setup.py                # Package setup
├── Dockerfile              # Main Dockerfile
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # Project documentation
```

## 📊 MLOps Pipeline

Loom.AI uses a robust MLOps pipeline for model training, validation, and deployment:

1. **Data Management**: DVC tracks data and model versions
2. **Experiment Tracking**: MLflow tracks experiments, metrics, and artifacts
3. **CI/CD**: GitHub Actions automates testing and deployment
4. **Deployment**: Models are served via FastAPI on Cloud Run or GKE

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Fast API](https://fastapi.tiangolo.com/) - API framework
- [Streamlit](https://streamlit.io/) - Frontend framework
- [DVC](https://dvc.org/) - Data version control
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Metaflow](https://metaflow.org/) - Data science pipeline framework 
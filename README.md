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
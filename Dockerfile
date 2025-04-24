# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/content /app/data/styles /app/data/processed

# Set environment variables for the application
ENV API_ENDPOINT=http://localhost:8000

# Expose port for the API
EXPOSE 8000

# Expose port for Streamlit
EXPOSE 8501

# Default command (can be overridden by docker-compose.yml or command line)
CMD ["python", "-m", "api.main"] 
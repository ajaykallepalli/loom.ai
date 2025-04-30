# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory
WORKDIR /app

# Optional: Install system dependencies if needed by your specific model or libraries
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     <your-dependencies> \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Consider using --default-timeout=100 or --no-cache-dir depending on your network/build environment
RUN pip install -r requirements.txt

# Copy the source code into the container
# This includes inference.py, style_transfer.py, and app.py
COPY src/ /app/src/

# Copy the pre-trained models
COPY models/ /app/models/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for the port (optional, Uvicorn uses --port)
# ENV PORT=8080

# Command to run the Uvicorn server for the FastAPI app
# It looks for the 'app' instance within the 'src.inference' module
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8080"] 
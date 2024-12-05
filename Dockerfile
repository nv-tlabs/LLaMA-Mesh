FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Gradio
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]

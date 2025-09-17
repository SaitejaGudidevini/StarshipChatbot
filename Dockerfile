FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY visualizer_fastapi.py .
COPY templates/ templates/

# Create output directory
COPY output/ output/

# Expose port
EXPOSE 8000

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["uvicorn", "visualizer_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files for StarshipChatbot Unified Server
COPY starship_unified.py .
COPY json_chatbot_engine.py .
COPY langgraph_chatbot.py .
COPY browser_agent_runner.py .

# Copy essential JSON data files
COPY CSU_Progress.json .
COPY browser_agent_test_output.json .

# Copy environment file (if exists)
COPY .env* ./ 2>/dev/null || true

# Create data directory for persistent storage (Railway volume will mount here)
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATA_DIR=/app/data
ENV JSON_DATA_PATH=CSU_Progress.json

# Healthcheck to ensure server is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/api/health').read()"

# Run the StarshipChatbot Unified Server
CMD uvicorn starship_unified:app --host 0.0.0.0 --port ${PORT:-8000}
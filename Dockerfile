FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (needed for sentence-transformers and numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only required Python application files
COPY starship_unified.py .
COPY json_chatbot_engine.py .
COPY langgraph_chatbot.py .
COPY browser_agent_runner.py .

# Copy required files from WorkingFiles (not the whole 9GB folder!)
RUN mkdir -p WorkingFiles
COPY WorkingFiles/__init__.py ./WorkingFiles/
COPY WorkingFiles/browser_agent.py ./WorkingFiles/
COPY WorkingFiles/hierarchical_crawler.py ./WorkingFiles/
COPY WorkingFiles/labeling.py ./WorkingFiles/

# Copy essential JSON data files (will be moved to /app/data on first startup)
COPY CSU_Progress.json .
COPY browser_agent_test_output.json .
COPY MelindaFile.json .

# Copy pre-built pickle caches if they exist (speeds up first startup)
# Note: If these don't exist locally, Docker build will skip them (won't fail)
# Generate them locally first with: python -c "from json_chatbot_engine import JSONChatbotEngine; JSONChatbotEngine('CSU_Progress.json')"
COPY *_qa_cache.pkl ./

# Copy frontend build (Bolt.new generated React app)
# The backend automatically serves this from frontend/dist/
COPY frontend/dist ./frontend/dist

# Copy environment file if it exists (optional - Railway/Render use dashboard env vars)
# Using wildcard pattern makes it non-fatal if file doesn't exist
COPY .env* ./

# Create data directory for persistent storage (Railway/Render volume will mount here)
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables for deployment
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATA_DIR=/app/data
ENV JSON_DATA_PATH=CSU_Progress.json

# Healthcheck to ensure server is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/api/health').read()"

# Run the StarshipChatbot Unified Server
CMD uvicorn starship_unified:app --host 0.0.0.0 --port ${PORT:-8000}
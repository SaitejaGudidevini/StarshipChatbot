FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal - no Playwright dependencies needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files for langgraph server
COPY langgraph_chatbot.py .
COPY langgraph_server.py .
COPY browser_agent_test_output.json .
COPY .env* ./

# Create data directory for Railway volume mount
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8000

# Healthcheck to ensure server is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/api/health').read()"

# Run the langgraph server application
CMD uvicorn langgraph_server:app --host 0.0.0.0 --port ${PORT:-8000}
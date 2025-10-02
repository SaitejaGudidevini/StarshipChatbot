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

# Copy application files for chatbot
COPY chatbot_server.py .
COPY vector_query_service.py .
COPY content_extractor.py .
COPY query_chroma.py .
COPY templates/ templates/

# Copy ChromaDB and output data
COPY chroma_db/ chroma_db/
COPY output/ output/

# Install Playwright and dependencies
RUN pip install playwright
RUN playwright install chromium

# Expose port
EXPOSE 8002

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8002

# Run the chatbot application
CMD ["uvicorn", "chatbot_server:app", "--host", "0.0.0.0", "--port", "8002"]
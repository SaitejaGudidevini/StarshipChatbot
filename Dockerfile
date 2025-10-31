FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Playwright browser dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    wget \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
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

# Install Playwright browsers (playwright should already be installed from requirements.txt)
RUN playwright install chromium

# Expose port
EXPOSE 8002

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8002

# Run the chatbot application
CMD ["uvicorn", "chatbot_server:app", "--host", "0.0.0.0", "--port", "8002"]
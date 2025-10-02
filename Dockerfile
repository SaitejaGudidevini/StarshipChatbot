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
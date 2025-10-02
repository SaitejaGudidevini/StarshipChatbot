FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Playwright browser dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
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
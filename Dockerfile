# Use base image (though it doesn't contain expected packages)
FROM chrisxlchen/pytorch-sentence-transformer:latest

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install all dependencies
COPY requirements-base.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-base.txt \
    && rm -rf /root/.cache/pip \
    && find /usr/local -type f -name "*.pyc" -delete \
    && find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy only necessary application files
COPY src/ ./src/
COPY app.py streamlit_app.py ./
COPY .env.example ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/documents /app/chroma_db && \
    chmod 755 /app/documents /app/chroma_db

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

EXPOSE 8000 8501

CMD ["python", "app.py"]
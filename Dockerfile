# CAD ML Platform - Optimized Multi-Stage Dockerfile
# Supports both CPU and GPU builds

# ================================
# Stage 1: Base dependencies
# ================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ================================
# Stage 2: Build dependencies
# ================================
FROM base as builder

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# ================================
# Stage 3: Production image
# ================================
FROM python:3.10-slim as production

# Labels
LABEL maintainer="CAD ML Platform Team" \
    version="1.0.0" \
    description="CAD ML Platform - Intelligent CAD Analysis Service"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/root/.local/bin:$PATH \
    # Application settings
    APP_ENV=production \
    APP_PORT=8000 \
    # Reduce memory footprint
    MALLOC_TRIM_THRESHOLD_=100000

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/models \
    && chown -R appuser:appuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ================================
# Stage 4: Development image
# ================================
FROM production as development

USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-warn-script-location -r requirements-dev.txt

# Copy tests
COPY --chown=appuser:appuser tests/ ./tests/

USER appuser

# Override command for development
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Stage 5: GPU image (optional)
# ================================
FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04 as gpu-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy from builder
COPY --from=builder /root/.local /root/.local

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

ENV PYTHONPATH=/app \
    PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

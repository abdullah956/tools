# Multi-Stage Dockerfile for Management Service
# Supports: development, staging, production
# Build with: docker build --target <stage> -t management:<stage> -f management/Dockerfile .

# ============================================
# Stage 1: Builder (installs dependencies)
# ============================================
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

WORKDIR /build

# Install build dependencies and Poetry
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY management/pyproject.toml management/poetry.lock ./
RUN poetry lock && poetry install --only main --no-interaction --no-ansi

# ============================================
# Stage 2: Base (shared runtime setup)
# ============================================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=management.configurations.settings \
    PYTHONPATH="/app"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories
RUN mkdir -p /app/uploads /app/static /app/media /app/staticfiles

# Copy application code
COPY management/ /app/management/
COPY envs/ /app/envs/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser
WORKDIR /app/management
EXPOSE 8000

# ============================================
# Stage 3: Development
# ============================================
FROM base AS development

ENV ENV=development

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# ============================================
# Stage 4: Staging
# ============================================
FROM base AS staging

ENV ENV=staging

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["gunicorn", "management.configurations.wsgi:application", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "2", \
    "--timeout", "300", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]

# ============================================
# Stage 5: Production
# ============================================
FROM base AS production

ENV ENV=production

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["gunicorn", "management.configurations.wsgi:application", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "4", \
    "--timeout", "120", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]

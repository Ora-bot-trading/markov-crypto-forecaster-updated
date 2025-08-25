# Multi-stage Dockerfile para markov-crypto-forecaster

# =============================================================================
# Stage 1: Base image con dependencias del sistema
# =============================================================================
FROM python:3.11-slim as base

# Metadata
LABEL maintainer="Senior Quant Engineer"
LABEL description="Markov Crypto Forecaster - Sistema de trading cuantitativo"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    make \
    # TA-Lib dependencies
    wget \
    tar \
    # System utilities
    curl \
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Instalar TA-Lib
# =============================================================================
FROM base as talib-builder

WORKDIR /tmp

# Descargar e instalar TA-Lib
ENV TALIB_VERSION=0.4.0
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-${TALIB_VERSION}-src.tar.gz \
    && tar -xzf ta-lib-${TALIB_VERSION}-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/ta-lib*

# Actualizar library path
RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/talib.conf && ldconfig

# =============================================================================
# Stage 3: Python dependencies
# =============================================================================
FROM talib-builder as python-deps

# Crear usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Crear directorio de aplicación
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 4: Application
# =============================================================================
FROM python-deps as application

# Copiar código fuente
COPY src/ ./src/
COPY config/ ./config/
COPY README.md LICENSE ./

# Crear directorios necesarios
RUN mkdir -p data/{raw,processed,features,models,signals,backtests} \
    && mkdir -p logs reports mlruns \
    && chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Variables de entorno para la aplicación
ENV PYTHONPATH=/app \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    MODELS_DIR=/app/data/models \
    CONFIG_DIR=/app/config

# Puerto para aplicaciones web (si se implementa)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# =============================================================================
# Stage 5: Production (default)
# =============================================================================
FROM application as production

# Punto de entrada por defecto
ENTRYPOINT ["python", "-m", "src.cli.main"]

# Comando por defecto
CMD ["--help"]

# =============================================================================
# Stage 6: Development
# =============================================================================
FROM application as development

# Instalar dependencias de desarrollo
RUN pip install --no-cache-dir \
    jupyter \
    jupyter-lab \
    ipykernel \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy

# Jupyter configuration
RUN mkdir -p /home/appuser/.jupyter \
    && echo "c.NotebookApp.token = ''" > /home/appuser/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password = ''" >> /home/appuser/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> /home/appuser/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/appuser/.jupyter/jupyter_notebook_config.py

# Exponer puerto de Jupyter
EXPOSE 8888

# Comando por defecto para desarrollo
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Ejemplos de uso:
# =============================================================================

# Build production image:
# docker build --target production -t markov-crypto-forecaster:latest .

# Build development image:
# docker build --target development -t markov-crypto-forecaster:dev .

# Run backtest:
# docker run -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data \
#   markov-crypto-forecaster:latest backtest --config config/binance_spot_example.yaml

# Run development environment:
# docker run -p 8888:8888 -v $(pwd):/app markov-crypto-forecaster:dev

# Run with custom entrypoint:
# docker run -it --entrypoint /bin/bash markov-crypto-forecaster:latest

# =============================================================================
# Docker Compose example:
# =============================================================================

# version: '3.8'
# services:
#   markov-forecaster:
#     build: 
#       context: .
#       target: production
#     volumes:
#       - ./config:/app/config:ro
#       - ./data:/app/data
#       - ./logs:/app/logs
#     environment:
#       - LOG_LEVEL=INFO
#       - PYTHONPATH=/app
#     command: backtest --config config/binance_spot_example.yaml
#   
#   jupyter:
#     build:
#       context: .
#       target: development
#     ports:
#       - "8888:8888"
#     volumes:
#       - .:/app
#     environment:
#       - JUPYTER_ENABLE_LAB=yes

# =============================================================================
# Optimizaciones para producción:
# =============================================================================

# 1. Multi-stage build para imagen más pequeña
# 2. Usuario no-root para seguridad
# 3. Cache de capas Docker optimizado
# 4. Health checks incluidos
# 5. Variables de entorno configurables
# 6. Volúmenes para persistencia de datos
# 7. Modo desarrollo separado con Jupyter
# 8. Dependencias del sistema minimizadas

# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (Docker cache layer) ───────────────────────────
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy entire project ────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ───────────────────────────────────────────
RUN mkdir -p models data

# ── Hugging Face Spaces runs as non-root user (UID 1000) ──────────────────
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# ── Expose Streamlit port ──────────────────────────────────────────────────
EXPOSE 7860

# ── Streamlit config for HF Spaces ────────────────────────────────────────
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Run the app ────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

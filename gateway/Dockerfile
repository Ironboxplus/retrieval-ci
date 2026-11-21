# retrival gateway (BWH app-tier) — auth + billing + reverse-proxy to the 4090.
# Build context = this gateway/ dir (small; keeps .venv/frontend/data out).
#   docker build -t ghcr.io/ironboxplus/retrival-gateway:latest gateway/
# Runtime config via env (GATEWAY_*); see deploy/bwh/gateway.env.example.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GATEWAY_DB_URL=sqlite:////data/gateway.db

WORKDIR /app

# deps first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# the gateway package → /app/gateway (so `gateway.app:app` imports)
COPY . /app/gateway/

# non-root + a writable data dir for the SQLite ledger (mount a volume here)
RUN useradd -m -u 10001 app && mkdir -p /data && chown -R app /app /data
USER app
VOLUME ["/data"]

EXPOSE 8001
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8001"]

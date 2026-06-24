FROM python:3.10-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user --timeout 600 torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user --timeout 600 -r requirements.txt

FROM python:3.10-slim AS runtime
WORKDIR /app
RUN groupadd -g 10001 appgroup && useradd -u 10000 -g appgroup -m -s /bin/bash appuser
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Note: 'quantize/' and 'README.md' have been removed from this list
COPY --chown=appuser:appgroup public/ ./public/
COPY --chown=appuser:appgroup backend/ ./backend/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup training/ ./training/
COPY --chown=appuser:appgroup export/ ./export/
COPY --chown=appuser:appgroup inference/ ./inference/
COPY --chown=appuser:appgroup config.py ./

RUN mkdir -p outputs data && chown -R appuser:appgroup /app
USER appuser
EXPOSE 8000
ENTRYPOINT ["python", "backend/server.py"]
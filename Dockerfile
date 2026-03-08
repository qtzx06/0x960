FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY train/ train/

RUN uv pip install --system --no-cache ".[openenv]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "zero960_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "300", "--ws-ping-timeout", "300"]

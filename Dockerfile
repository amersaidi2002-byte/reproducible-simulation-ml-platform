FROM python:3.12.9-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIMML_BUNDLE=/app/artifacts/bundle
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/api.lock requirements/api.lock
RUN python -m pip install --no-cache-dir -r requirements/api.lock
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home appuser
COPY artifacts/bundle/ artifacts/bundle/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["python", "-m", "uvicorn", "simml.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

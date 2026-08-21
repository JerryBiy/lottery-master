FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DLT_ENV=production \
    DLT_HOST=0.0.0.0 \
    DLT_PORT=5000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/models /app/reports \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 5000
VOLUME ["/app/data", "/app/models", "/app/reports"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/v1/ready', timeout=3)"

CMD ["python", "run_web.py"]

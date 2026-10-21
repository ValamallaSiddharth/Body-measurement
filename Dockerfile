FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=5000 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
# MediaPipe requires opencv-contrib-python in addition to the headless wheel.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt README.md ./
COPY bodymeasure ./bodymeasure
RUN pip install --no-cache-dir '.[server]' && useradd --uid 10001 --create-home appuser
COPY gunicorn.conf.py ./
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "bodymeasure.web:app"]

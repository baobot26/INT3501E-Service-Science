FROM python:3.12-slim

WORKDIR /app

# rasterio is kept in requirements-dev.txt for the optional data-fetch job.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY pytest.ini ./
COPY src/ src/
COPY tests/ tests/
COPY app.py ./
COPY data/ data/

RUN useradd -m runner && chown -R runner:runner /app
USER runner

# data/ and results/ are bind-mounted at runtime; tests run offline by default
CMD ["pytest", "-q"]

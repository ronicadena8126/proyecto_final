# Imagen de los trabajos de producción (simulación, monitoreo, reentrenamiento). Kestra la ejecuta
# con el Docker task runner (pullPolicy: NEVER) en la red del proyecto, para llegar a la API y a MLflow.
#   docker compose build jobs
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-jobs.txt docker/requirements-jobs.txt
RUN pip install --no-cache-dir -r docker/requirements-jobs.txt
COPY src/ src/
COPY monitoreo/ monitoreo/
COPY scripts/ scripts/
COPY api/ api/
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
ENTRYPOINT []
CMD ["python", "-c", "print('imagen de trabajos lista')"]

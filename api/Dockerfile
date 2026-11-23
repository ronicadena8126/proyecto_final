# Imagen de la API de predicción. El modelo NO va dentro de la imagen: se descarga de MLflow
# (models:/anticipar-mora-modelo@champion) al arrancar y cada vez que cambia el alias, así que
# promover o hacer rollback de un modelo no requiere reconstruir ni volver a publicar la imagen.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt
COPY src/ src/
COPY api/ api/
ENV PYTHONUNBUFFERED=1 REGISTRO_DB=/app/registro/registro_api.db
RUN useradd --create-home --uid 1000 app && mkdir -p /app/registro && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD curl -fsS http://localhost:8000/salud || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

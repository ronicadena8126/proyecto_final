# MLflow 2.17.2 (misma versión que el requirements.txt del curso) + driver de Postgres.
FROM ghcr.io/mlflow/mlflow:v2.17.2
RUN pip install --no-cache-dir psycopg2-binary==2.9.9

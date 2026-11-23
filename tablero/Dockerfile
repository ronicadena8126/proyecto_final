FROM python:3.11-slim
WORKDIR /app
COPY tablero/requirements.txt tablero/requirements.txt
RUN pip install --no-cache-dir -r tablero/requirements.txt
COPY src/__init__.py src/almacen.py src/
COPY tablero/ tablero/
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
EXPOSE 8501
CMD ["streamlit", "run", "tablero/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]

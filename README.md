# Anticipar la mora en tarjetas de crédito: proyecto final (PSets 1 a 4)

Este modelo predice, con un mes de anticipación, qué clientes de tarjeta de crédito no pagarán su próxima cuota. Cada mes, cobranza recibe una lista priorizada y llama antes del vencimiento.

El repositorio reúne todo el proyecto: la ingesta y el modelo dimensional (PSet 2), la experimentación (PSet 3), el despliegue, el monitoreo, el reentrenamiento y la comunicación (PSet 4).

- **Documento técnico:** `docs/PSet4_memo_Almeida_Cadena.pdf`
- **Presentación ejecutiva:** `docs/PSet4_presentacion_Almeida_Cadena.pdf`
- **Registro de correcciones:** `docs/cambios.md`
- **Video de la demo:** enlace en el documento técnico y en `docs/video_url.txt`

```
proyecto_final/
├── docker-compose.yml        Kestra, Spark, MLflow, API y tablero (+ imágenes que corre Kestra)
├── .env.example              variables de entorno (copiar a .env; nunca subir .env)
├── requirements.txt          entorno de desarrollo (Python 3.11)
├── .github/workflows/        ci.yml (pytest + build + humo) y cd.yml (imagen a GHCR con el hash del commit)
├── kestra/                   ingesta.yml, produccion_lotes.yml, monitoreo.yml, reentrenamiento.yml
├── dbt/  spark/              PSet 2: RAW → BRONZE → SILVER → GOLD (dbt + 72 tests) y OBT (Spark)
├── src/                      datos, pipeline, negocio, tracking (PSet 3) + almacen, contrato, registro,
│                             simulacion, produccion, reentrenar, exportar_fixture (PSet 4)
├── notebooks/                01–05 (PSet 3, ya ejecutados) · 06 simulación · 07 análisis del monitoreo
├── api/                      FastAPI: main.py, esquemas.py (Pydantic), Dockerfile
├── monitoreo/                deriva.py (PSI, KS, chi²) y alertas.py (métricas, alertas, disparador)
├── tablero/                  Streamlit
├── tests/                    contrato, modelo, comportamiento, respaldo + fixtures/ (copia del champion)
├── scripts/                  encode_secrets.sh, deploy_kestra.sh, evidencia_operacion.py, simular_local.py
└── docs/                     arquitectura, modelo dimensional, cambios.md, CSV de evidencia, PDFs e historial/
```

## 1. Requisitos y configuración

1. **Herramientas:** Docker Desktop (con 8 GB de RAM o más), Python 3.11 (`uv`) y Google Chrome (para generar los PDF). En Mac también hace falta `brew install libomp`.

2. **Snowflake:** se usan la base `PSET2_MORA`, el usuario de servicio `PSET2_SVC` y la autenticación key-pair del PSet 2 (`docs/snowflake_setup.sql`). El esquema `PRODUCCION` se crea solo.

3. **Variables de entorno.** Copia la plantilla y completa los valores. `SNOWFLAKE_PRIVATE_KEY_PATH` debe ser una ruta **absoluta** y `ADMIN_TOKEN` debe ser largo y aleatorio.
   ```bash
   cp .env.example .env
   bash scripts/encode_secrets.sh       # genera .env_encoded (secretos de Kestra, en base64)
   ```

4. **Entorno de Python:**
   ```bash
   uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
   ```

## 2. Levantar la infraestructura (un solo comando)

Antes, detén los stacks de los PSets 2 y 3 si están corriendo: usan los mismos puertos.

```bash
docker compose up -d --build
```

Servicios que quedan disponibles:
- Kestra: http://localhost:8080
- Spark: http://localhost:8082
- MLflow: http://localhost:5001
- API: http://localhost:8000/docs
- Tablero: http://localhost:8501

MLflow reutiliza los volúmenes del PSet 3 (`pset3_mlflow-db` y `pset3_mlflow-artifacts`), así que conserva todas las corridas y el Model Registry. Para publicar los flujos de Kestra:

```bash
bash scripts/deploy_kestra.sh
```

La API usa la imagen que publica `cd.yml` en GitHub Container Registry (`API_TAG` en `.env`). Para actualizarla:

```bash
docker compose pull api && docker compose up -d api
```

Si la imagen no está disponible, `docker compose up --build` la construye en local con `api/Dockerfile`.

## 3. Orden de ejecución completo (desde cero)

| # | Paso | Cómo | Resultado |
|---|---|---|---|
| 1 | **Fuente → RAW → BRONZE → SILVER → GOLD → OBT** | Kestra → `usfq.proyecto_final.ingesta`, con `meses = 2005-04 2005-05 2005-06 2005-07 2005-08 2005-09` (backfill) | `PSET2_MORA.OBT.OBT_CUSTOMER_DEFAULT` (30.000 filas) |
| 2 | **EDA y modelo** | notebooks `01_eda` → `05_seleccion` (+ `03b_llm_prompting`, `05b_reproducir_champion`) | Corridas en MLflow; `@champion` / `@challenger` |
| 3 | **Registro → fixture de CI** | `python -m src.exportar_fixture` (con el champion del PSet 3, **antes** de simular) | `tests/fixtures/` + métrica `auc_muestra_fija` en MLflow |
| 4 | **Pruebas** | `pytest` (en GitHub Actions corren solas en cada push / PR) | 21 pruebas en verde |
| 5 | **API** | `docker compose up -d api` · ejemplo abajo | `/salud`, `/modelo`, `/predecir` |
| 6 | **Simulación: lotes** | notebook `06_simulacion_produccion` | `PRODUCCION.LOTES` y `PRODUCCION.TARGETS` (10 lotes, deriva desde el lote 6) |
| 7 | **Simulación + monitoreo + reentrenamiento** | Kestra → `produccion_lotes` (lotes 1–10, `cierre = true`). En cada pausa de `reentrenamiento`: Resume con `aprobar` y un comentario | `PREDICCIONES`, `MONITOREO`, `ALERTAS`, `REENTRENAMIENTOS` |
| 8 | **Rollback e interruptor** | `python scripts/evidencia_operacion.py` | `docs/evidencia_operacion.json` |
| 9 | **Análisis y evidencia** | notebook `07_analisis_monitoreo` | `docs/*.csv`, gráficos y `docs/resumen_simulacion.json` |
| 10 | **Documento y presentación** | `python docs/generar_documento.py && python docs/generar_presentacion.py` | PDFs en `docs/` |

Los notebooks 01–05 del PSet 3 se entregan **ya ejecutados**, con sus salidas. Volver a correrlos crea versiones nuevas en el Model Registry y cambia el champion. Para reproducirlos desde cero, hazlo antes del paso 3.

### Ejemplo de uso de la API

```bash
curl -s localhost:8000/modelo
curl -s -X POST localhost:8000/predecir -H 'Content-Type: application/json' \
     -d "$(python -c 'from api.esquemas import Cliente; import json; print(json.dumps(Cliente.model_config["json_schema_extra"]["example"]))')"
```

### Operación

- **Rollback** sin reconstruir la imagen: `python -m src.registro rollback`. Vuelve `@champion` a `@anterior` y la API lo toma en 30 segundos o menos.
- **Estado del registro:** `python -m src.registro estado`
- **Interruptor de respaldo:** `curl -X POST localhost:8000/admin/respaldo -H "X-Token: $ADMIN_TOKEN" -H 'Content-Type: application/json' -d '{"activo": true, "motivo": "..."}'`. También se activa con `MODO_RESPALDO=1` en `.env`.
- **Simulación sin Kestra** (depuración): `python scripts/simular_local.py`. Aprueba o rechaza según la regla, sin pausa humana.

## 4. Pruebas

```bash
pytest -v
```

La suite cubre cuatro grupos:
- **Contrato:** 200, 422 con el campo que falló, y esquema igual a la firma de MLflow.
- **Modelo:** reproduce el AUC registrado sobre una muestra fija, con tolerancia 1e-3.
- **Comportamiento de negocio.**
- **Respaldo:** modelo que no carga, interruptor por endpoint e interruptor por variable de entorno.

En CI se usa la copia del champion de `tests/fixtures/`, porque allí no hay MLflow.

## 5. Seguridad

`.env`, `.env_encoded`, `*.p8` y `data/` están en `.gitignore`. La API no guarda credenciales de Snowflake: Kestra carga su registro de predicciones. Los endpoints `/admin/*` exigen `X-Token`.

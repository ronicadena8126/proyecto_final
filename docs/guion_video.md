# Guion del video de la demo (máximo 7 minutos)

Grabar con QuickTime o con `Cmd + Shift + 5`, pantalla completa, con el micrófono encendido. Subirlo a YouTube como **no listado** (o a Google Drive con acceso por enlace) y pegar el enlace en `docs/video_url.txt`.

Antes de grabar, deja preparado:
- `docker compose down` (para mostrar el arranque desde cero).
- Una terminal en `~/proyecto_final` con el venv activo.
- Pestañas abiertas: Kestra, MLflow, la API en `/docs` y el tablero.

| Tiempo | Qué mostrar | Qué decir (idea) |
|---|---|---|
| 0:00–0:45 | `docker compose up -d --build`, luego `docker compose ps` con los servicios arriba | "Toda la infraestructura levanta con un comando: Kestra, Spark, MLflow, la API y el tablero." |
| 0:45–2:00 | Kestra → `ingesta` → Execute con los 6 meses → Gantt: `ingesta` → `dbt_build` → `spark_obt` en verde. En Snowflake, la tabla OBT con 30.000 filas | "Carga idempotente por mes con retry; dbt construye Bronze, Silver y Gold con 72 tests, y Spark arma la OBT." (Si tarda, acelera esa parte del video.) |
| 2:00–2:50 | API `/docs`: `GET /modelo` (versión y run_id), `POST /predecir` válido (probabilidad, decisión y acción), y la misma solicitud con `education_code: 9` → **422** con el campo | "La API carga el champion por alias, valida con Pydantic y aplica el umbral de capacidad." |
| 2:50–3:30 | Terminal: `python -m src.registro rollback`, luego `curl localhost:8000/modelo` (otra versión). Restituir: `python -m src.registro asignar --version <la de antes>` | "El rollback es mover el alias en MLflow; la imagen no se reconstruye." |
| 3:30–4:30 | Kestra → ejecución de `produccion_lotes` → subflujo `monitoreo` del lote 6 → log con **ALERTAS** en rojo (PSI de `limit_bal` y de las utilizaciones). Comparar con el lote 3: "Sin alertas" | "El lote con deriva dispara la alerta; los lotes limpios no." |
| 4:30–5:50 | Subflujo `reentrenamiento` en **PAUSA** → log `comparacion` (regla CUMPLE) → Resume con `aprobar = true` y un comentario → `decidir` → en MLflow, los alias `@champion` en la versión nueva y `@anterior`; `/modelo` muestra la versión nueva | "El reentrenamiento compara champion y challenger en un lote que ninguno vio y espera aprobación humana. Al aprobar, la API cambia sola." |
| 5:50–6:30 | `curl -X POST localhost:8000/admin/respaldo -H "X-Token: …" -d '{"activo":true,"motivo":"demo"}'`, luego `POST /predecir` → `fuente: regla_negocio, respaldo: true`. Desactivar | "El interruptor vuelve a la regla de negocio sin tocar el modelo." |
| 6:30–7:00 | Tablero: ahorro por lote frente a la regla, AUC frente al umbral, mapa de calor de PSI, reentrenamientos marcados | "Resultado: el modelo evita más pérdida que la regla y se recupera del cambio de mercado." |

**Consejo.** La simulación completa con Snowflake tarda varios minutos. Ejecútala antes de grabar y, en el video, muestra las ejecuciones ya terminadas en Kestra. Para la pausa en vivo, deja un reentrenamiento pendiente lanzando a mano el flujo `reentrenamiento` con `lote = 10` y `motivo = manual`, y apruébalo o recházalo durante la grabación.

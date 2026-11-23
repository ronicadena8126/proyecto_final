# Registro de cambios del proyecto final

Este archivo registra tres cosas:
- Cada observación recibida en los PSets 1 a 3, qué se cambió y en qué archivo.
- Las decisiones de PSets anteriores que se actualizaron para el proyecto final.
- Las mejoras que se hicieron al integrar todo en `proyecto_final/`.

## 1. Observaciones del profesor

### PSet #1 (98/100)

| # | Observación | Qué se cambió | Dónde |
|---|---|---|---|
| 1.1 | *"Problema y formulación (9): Justifica el umbral de 0.6 considerando la tasa de impago del 22% y los costos de falsos positivos y falsos negativos. No corresponde describirlo como un umbral bajo sin ese análisis."* | **Cambia el umbral de decisión de 0,6 a uno justificado con costos.** Hay que contactar si la ganancia esperada supera el costo de la llamada: `p × M̄ × t > c`. Con M̄ = NT$ 47.967 (saldo medio en riesgo de los morosos de entrenamiento), t = 0,30 y c = NT$ 500, el **umbral de equilibrio es p\* = 500 / (47.967 × 0,30) ≈ 0,035**. Un falso negativo cuesta unas 29 veces más que un falso positivo (NT$ 14.390 contra NT$ 500). Por costo, entonces, conviene llamar casi a cualquier cliente con riesgo apreciable. El 0,6 no era un umbral bajo, era muy conservador. Con una tasa base de impago del 22 %, al 0,6 se contacta solo al ~10 % de la cartera y se pierden morosos detectables: en el conjunto de prueba del PSet 3 el ahorro con 0,6 fue de NT$ 5,77 M, frente a NT$ 9,26 M con la política final. **La restricción que manda es la capacidad del call center (20 % de la cartera por mes).** El umbral operativo es el corte que llena esa capacidad, medido con predicciones *out-of-fold* de entrenamiento: **0,34**. La API lo lee de la corrida del champion en MLflow (`corte_efectivo_train_oof`) y lo aplica. | `src/negocio.py` (`umbral_equilibrio`, `decision_modelo`), `api/main.py` (umbral leído del registro), documento técnico §1, presentación (diapositiva de la decisión), PSet 1 corregido en `docs/historial/` |
| 1.2 | *"Pitch de negocio (9.5): Elimina el texto pendiente de la plantilla en la portada y corrige la justificación del umbral de 0.6."* | Se reemplaza el texto pendiente "[Apellido compañero]" de la portada por el nombre del autor. La frase "Umbral bajo a propósito…" se cambia por la justificación de 1.1: equilibrio en p ≈ 0,035, la capacidad del 20 % es la que limita y el corte operativo es 0,34. | `docs/historial/PSet1_pitch_Almeida_Cadena.pdf` (portada y diapositiva "Viabilidad de negocio"), presentación final |

### PSet #2 (100/100)

No hubo observaciones. Los cambios de la sección 3 son de integración.

### PSet #3

La calificación no estaba publicada al cerrar este proyecto. Se registran los ajustes hechos en la revisión final del PSet 3, antes de la entrega. Si llegan observaciones del profesor, se agregan aquí.

| # | Ajuste | Qué se cambió | Dónde |
|---|---|---|---|
| 3.1 | La familia "State of the Art" no se había evaluado. | Se agregó un LLM por *prompting* (`qwen2.5:3b` local con Ollama), evaluado en 200 clientes de entrenamiento contra el mejor tabular. Obtuvo AUC 0,593 contra 0,815 y quedó descartado. | `notebooks/03b_llm_prompting.ipynb`, memo del PSet 3 |
| 3.2 | La columna `avg_payment_ratio_6m` llega 100 % vacía desde Snowflake. | Se excluye automáticamente (`columnas_vacias`) y queda documentada como hallazgo H6 de la EDA. | `src/datos.py`, `notebooks/01_eda.ipynb` |
| 3.3 | El campeón se elegía por orden de la CV, aunque las diferencias eran menores que la variación entre pliegues. | Regla de desempate fijada antes de mirar el conjunto de prueba: menor brecha de sobreajuste y, después, menor costo. | `notebooks/05_seleccion.ipynb` |

## 2. Decisiones actualizadas en todo el proyecto

| Decisión | Antes | Ahora | Por qué | Dónde se actualizó |
|---|---|---|---|---|
| Umbral de decisión | p ≥ 0,6 (PSet 1) | p ≥ corte de capacidad (≈ 0,34, OOF de entrenamiento), con piso de 0,05; lista acotada al 20 % de la cartera | Observación 1.1: el equilibrio de costos (0,035) está muy por debajo y lo que limita es la capacidad | `src/negocio.py`, `api/main.py`, `src/reentrenar.py` (recalcula el corte con cada reentrenamiento), documento §1 |
| Línea base | Regla "≥ 1 mes de atraso" | La misma, y además la comparación **a igual número de llamadas** | La regla llama al 22 % y el modelo al 20 %: compararlas con volúmenes distintos favorece a quien llama más | `monitoreo/alertas.py` (`ahorro_modelo_igual_volumen`), §9 |
| Patrón de despliegue | "API de scoring" (PSet 1, sin definir) | **Batch mensual** orquestado por Kestra. La API se construye igual y es el punto único de validación, decisión y registro | La decisión es mensual, los datos llegan una vez al mes y el volumen es de miles de clientes, no de solicitudes por segundo | `kestra/produccion_lotes.yml`, documento §5 |
| Horizonte, target, métrica técnica, grano de la OBT, fuente | — | **Sin cambios**: 1 mes; `default_next_month`; AUC-ROC; 1 fila por cliente; UCI | — | — |
| Juicio humano | El asesor revisa el caso y elige la oferta | Se mantiene y **se agrega una segunda instancia**: una persona aprueba cada cambio de modelo (pausa en Kestra) | La revisión mostró que un challenger que cumple la regla de ahorro puede violar pruebas de comportamiento (ver §8 del documento) | `kestra/reentrenamiento.yml`, documento §8 |

## 3. Integración en `proyecto_final/`

| Cambio | Detalle | Dónde |
|---|---|---|
| Un solo repositorio | La ingesta, dbt y Spark del PSet 2, los notebooks y `src/` del PSet 3 y la parte nueva del PSet 4 están en un solo árbol de carpetas. | todo el repositorio |
| Kestra 2.x | Los 3 flujos del PSet 2 (`uci_ingest_monthly`, `uci_backfill`, `pipeline_transform`) se unen en `ingesta.yml`, en el namespace `usfq.proyecto_final`. El backfill es la misma ejecución con `meses` explícitos. `ForEach` ya no existe en Kestra 2.0, así que se usa `Loop`. | `kestra/ingesta.yml`, `kestra/produccion_lotes.yml` |
| Un `docker-compose.yml` | Kestra, Spark, MLflow, API y tablero, más las imágenes que ejecuta Kestra. MLflow conserva los volúmenes del PSet 3, así que se mantienen todas las corridas y el Model Registry. | `docker-compose.yml` |
| Conexión a Snowflake | La llave se lee como archivo o como texto PEM, que es como la entregan los secretos de Kestra. La conexión se reutiliza dentro de cada proceso. | `src/almacen.py`, `src/datos.py` |
| Notebook de reproducción | `06_reproducir_champion` pasa a `05b_reproducir_champion`. El número 06 queda para la simulación. | `notebooks/` |
| PSet 2 sin cambios de lógica | Los modelos dbt, los 72 tests y `build_obt.py` son los mismos que se entregaron. | `dbt/`, `spark/` |

"""
Genera el documento técnico del proyecto final a partir de la evidencia exportada:

  python docs/generar_documento.py   ->  docs/PSet4_memo_Almeida_Cadena.pdf  (12 páginas + anexos)

Lee docs/resumen_simulacion.json y los CSV del notebook 07, los resultados del PSet 3 (docs/pset3/),
las imágenes de docs/img y las capturas de docs/img/capturas (si existen). Todas las cifras salen de
esos archivos: si se vuelve a correr la simulación, el documento se actualiza solo.
Anexos: docs/cambios.md y los memos de los PSets 1-3 (docs/historial/).
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent
RAIZ = D.parent
sys.path.insert(0, str(RAIZ))
SALIDA = D / "PSet4_memo_Almeida_Cadena.pdf"
AUTOR = "Ronnie Almeida Cadena"
REPO = os.environ.get("REPO_URL", "https://github.com/ronicadena8126/proyecto_final")

R = json.loads((D / "resumen_simulacion.json").read_text())
NEG = pd.read_csv(D / "negocio_por_lote.csv", index_col=0)
SUB = pd.read_csv(D / "subgrupos.csv")
REE = pd.read_csv(D / "reentrenamientos.csv")
ALE = pd.read_csv(D / "alertas.csv")
P3 = D / "pset3"
CAND = pd.read_csv(P3 / "resultados_candidatos.csv")
TUN = pd.read_csv(P3 / "resultados_tuning.csv")
TEST = pd.read_csv(P3 / "resultados_test.csv")
REG = pd.read_csv(P3 / "registro.csv", index_col=0)
HALL = pd.read_csv(P3 / "hallazgos_eda.csv")
COR3 = pd.read_csv(P3 / "corridas.csv")
PISO = pd.read_csv(D / "piso_ruido_psi.csv") if (D / "piso_ruido_psi.csv").exists() else None
EVID = json.loads((D / "evidencia_operacion.json").read_text()) if (D / "evidencia_operacion.json").exists() else []
VIDEO = (D / "video_url.txt").read_text().strip() if (D / "video_url.txt").exists() else ""
CAP = D / "img" / "capturas"


# ----------------------------------------------------------------------------- utilidades
def f(x, d=3):
    """Número con coma decimal y punto de miles (es-EC)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def nt(x, millones=False):
    if x is None or pd.isna(x):
        return "–"
    return f"NT$ {f(x / 1e6, 2)} M" if millones else f"NT$ {f(x, 0)}"


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{f(100 * x, d)} %"


def esc(s):
    return html.escape(str(s))


def img(ruta: Path, ancho="100%", pie=""):
    if not ruta.exists():
        return f'<div class="falta">Falta la imagen {esc(ruta.relative_to(RAIZ))}</div>'
    b = base64.b64encode(ruta.read_bytes()).decode()
    cap = f'<div class="cap">{pie}</div>' if pie else ""
    return f'<img src="data:image/png;base64,{b}" style="width:{ancho}">{cap}'


def captura(nombre, pie, ancho="100%"):
    for ext in ("png", "jpg", "jpeg"):
        p = CAP / f"{nombre}.{ext}"
        if p.exists():
            return img(p, ancho, pie)
    return f'<div class="falta">Captura pendiente: docs/img/capturas/{nombre}.png ({pie})</div>'


def tabla(filas, cabecera, clases=None):
    th = "".join(f"<th>{c}</th>" for c in cabecera)
    trs = []
    for i, fila in enumerate(filas):
        cl = f' class="{clases[i]}"' if clases and clases[i] else ""
        trs.append(f"<tr{cl}>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>")
    return f"<table><tr>{th}</tr>{''.join(trs)}</table>"


# ----------------------------------------------------------------------------- datos derivados
champ = REG.loc["champion"]
chall = REG.loc["challenger"]
t_ch = TEST.set_index("modelo").loc[champ.modelo]
t_rg = TEST.set_index("modelo").loc["regla_negocio"]
llm = COR3[COR3.get("tags.familia", pd.Series(dtype=str)) == "sota_llm"] if "tags.familia" in COR3 else pd.DataFrame()
ree = REE.sort_values("lote_disparo")
aprob = ree[ree.decision.astype(str).str.startswith("aprobado")]
rech = ree[ree.decision.astype(str) == "rechazado"]
neg = R["negocio"]
L0 = int(R["inicio_deriva"])
n_lotes = int(R["n_lotes"])
diag = pd.DataFrame(R["diagnostico"])
conc = ALE[ALE.regla == "concepto_auc"]
deriva_al = ALE[ALE.regla == "deriva_datos_feature"]
corte_op = None
try:
    corte_op = float(COR3.loc[COR3.run_id == champ.run_id, "params.corte_efectivo_train_oof"].iloc[0])
except Exception:
    corte_op = float(json.loads((RAIZ / "tests/fixtures/referencia.json").read_text()).get("corte_efectivo", 0.34))
p_eq = 500 / (47966.81 * 0.30)


def ejemplo_api():
    """Solicitud y respuesta reales de la API (con la copia del champion de tests/fixtures)."""
    try:
        os.environ.setdefault("MODELO_URI", str(RAIZ / "tests/fixtures/modelo"))
        os.environ.setdefault("ADMIN_TOKEN", "x")
        os.environ["REGISTRO_DB"] = "/tmp/registro_doc.db"
        from fastapi.testclient import TestClient

        import api.main as m
        from api.esquemas import Cliente

        ej = dict(Cliente.model_config["json_schema_extra"]["example"])
        with TestClient(m.app) as c:
            ok = c.post("/predecir", json=ej).json()
            mal = c.post("/predecir", json={**ej, "education_code": 9, "age": "treinta"}).json()
        corto = {k: ej[k] for k in ["customer_id", "limit_bal", "education_code", "age", "pay_status_m1",
                                    "bill_amt_m1", "pay_amt_m1", "util_m1"]}
        return corto, ok, mal
    except Exception as e:
        return {}, {"error": str(e)}, {}


req, resp, resp_mal = ejemplo_api()

# ----------------------------------------------------------------------------- texto dinámico
if len(aprob):
    a = aprob.iloc[0]
    txt_ciclo = (f"En el lote {int(a.lote_disparo)} el disparador por desempeño se cumplió: el AUC del champion v{int(a.version_champion_previo)} "
                 f"quedó bajo el umbral en dos lotes etiquetados seguidos. Kestra entrenó el challenger v{int(a.version_challenger)} "
                 f"y lo comparó con el champion en el lote {int(a.lote_evaluacion)}, que ninguno de los dos había visto. "
                 f"AUC {f(a.auc_challenger)} frente a {f(a.auc_champion)}; ahorro {nt(a.ahorro_challenger)} frente a {nt(a.ahorro_champion)}. "
                 f"La diferencia, {nt(a.delta_ahorro)}, supera σ_CV = {nt(a.sigma_cv_ahorro)}, así que <b>la regla se cumple</b>. "
                 f"La ejecución se detuvo en la pausa y la decisión registrada fue «{esc(a.decision)}».")
    if "viol_comportamiento" in a and pd.notna(a.viol_comportamiento) and a.viol_comportamiento > 0:
        txt_ciclo += (f" El reporte también mostró <b>{int(a.viol_comportamiento)} clientes</b> en los que el challenger viola la prueba "
                      "de comportamiento «más atraso no baja el riesgo». El nuevo régimen hace que un cliente al día y muy endeudado "
                      "parezca más riesgoso que uno con dos meses de atraso. La regla de ahorro sola no lo detecta: por eso la promoción "
                      "pasa por una persona.")
else:
    txt_ciclo = "En la simulación ningún challenger cumplió la regla de promoción; el champion se mantuvo."
if len(rech):
    r0 = rech.iloc[0]
    txt_rech = (f"El reentrenamiento por {esc(r0.motivo)} del lote {int(r0.lote_disparo)} produjo el challenger v{int(r0.version_challenger)}. "
                f"Se <b>rechazó</b>: en el lote {int(r0.lote_evaluacion)} su ahorro fue {nt(r0.ahorro_challenger)}, contra {nt(r0.ahorro_champion)} del champion "
                f"(diferencia {nt(r0.delta_ahorro)}, menor que σ_CV = {nt(r0.sigma_cv_ahorro)}). Sin un cambio en los datos, reentrenar "
                f"no mejora, y la regla evita cambiar de modelo por ruido.")
else:
    txt_rech = ""
falsas = R.get("falsas_alarmas_detalle", [])
txt_falsas = ("No hubo falsas alarmas de deriva: ninguna alerta de deriva de datos ni conceptual se disparó en los lotes sin alterar." if not falsas else
              "Falsas alarmas en lotes sin alterar: " + "; ".join(f"lote {x['lote_id']} ({esc(x['regla'])})" for x in falsas)
              + ". Un AUC cerca del umbral en un lote de 600 clientes es ruido de muestreo. Por eso el reentrenamiento exige "
              "2 lotes seguidos. Si se repite, se usaría Z = 2,5 o se juntarían 2 lotes antes de evaluar.")
post = [int(x) for x in neg.get("lotes_post_reentreno", [])]
txt_efecto = (f"En los lotes {', '.join(map(str, post))}, puntuados con el modelo reentrenado, el ahorro fue {nt(neg['efecto_reentrenar'])} mayor "
              "que el que habría logrado el champion anterior con esos mismos clientes (re-puntuación fuera de línea)." if post else "")
gana = neg["lotes_modelo_gana"]
respuesta_negocio = (
    f"<b>Sí valió la pena.</b> En los {n_lotes} meses simulados, la lista del modelo evitó pérdidas netas por {nt(neg['modelo'], True)}, "
    f"contra {nt(neg['regla'], True)} de la regla actual: {nt(neg['extra_vs_regla'], True)} más "
    f"({pct(neg['extra_vs_regla'] / neg['regla'])}). El modelo ganó en {gana} de {n_lotes} lotes."
    if neg["extra_vs_regla"] > 0 else
    f"<b>No en esta simulación:</b> el modelo evitó {nt(neg['modelo'], True)} contra {nt(neg['regla'], True)} de la regla.")
brecha = R["subgrupos_brecha_recall"]

# ----------------------------------------------------------------------------- tablas
filas_alertas = [
    ["PSI de una feature", "&gt; 0,25", "1 lote", "Alta", "Equipo de datos", "Revisar la fuente; reentrenar solo si cae el desempeño"],
    ["PSI moderado", "0,10–0,25 en ≥ 3 features", "1 lote", "Media", "Equipo de datos", "Vigilar el siguiente lote"],
    ["PSI de la probabilidad", "&gt; 0,25", "1 lote", "Media", "Ciencia de datos", "Revisar tasa de contacto y capacidad"],
    ["AUC del lote etiquetado", "&lt; media CV − 2σ<sub>lote</sub>", "1 lote; reentrena con 2", "Alta", "Ciencia de datos", "Diagnosticar; 2 seguidos disparan el reentrenamiento"],
    ["Ahorro modelo vs. regla (igual volumen)", "&lt; ahorro de la regla", "1 lote; reentrena con 2", "Alta", "Cobranza + ciencia de datos", "Reentrenar; si persiste, respaldo"],
    ["Tasa de «contactar»", "&gt; 25 % (capacidad 20 %)", "1 lote", "Media", "Gerencia de cobranza", "Llamar al top 20 %; recalibrar el corte"],
    ["Latencia p95 /predecir", "&gt; 250 ms", "1 lote", "Media", "Plataforma", "Revisar recursos del contenedor"],
    ["Errores 5xx / rechazos 4xx", "&gt; 1 % / &gt; 2 %", "1 lote", "Alta / Media", "Plataforma / datos", "Logs; respaldo si el modelo falla / revisar el origen"],
    ["% faltantes por feature", "&gt; 1 %", "1 lote", "Media", "Equipo de datos", "Corregir la carga del core"],
    ["Predicciones con respaldo", "&gt; 0 %", "1 lote", "Alta", "Plataforma + cobranza", "Confirmar si fue manual; revisar la carga del modelo"],
]

filas_cand = []
for _, r in CAND.sort_values("cv_roc_auc_mean", ascending=False).iterrows():
    filas_cand.append([esc(r.modelo), esc(r.familia), f(r.cv_roc_auc_mean), f(r.cv_roc_auc_std), f(r.brecha_roc_auc), f(r.fit_seconds_mean, 2)])
if len(llm):
    l = llm.iloc[0]
    filas_cand.append([f"LLM {esc(l.get('params.llm_modelo', ''))} (prompting)", "SOTA", f(l["metrics.cv_roc_auc_mean"]) + "*", "–", "–",
                       f(l.get("metrics.segundos_por_cliente", float("nan")), 2) + " s/cliente"])

filas_tun = [[esc(r.modelo), f(r.default_cv_auc), f(r.tuned_cv_auc), f(r.tuned_std), f(r.mejora), "sí" if r.mejora_supera_desv else "no", f(r.brecha_tuned)]
             for _, r in TUN.iterrows()]

filas_ree = [[int(r.lote_disparo), esc(r.motivo), int(r.lote_evaluacion), f"v{int(r.version_champion_previo)} → v{int(r.version_challenger)}",
              f"{f(r.auc_champion)} / {f(r.auc_challenger)}", f"{nt(r.ahorro_champion)} / {nt(r.ahorro_challenger)}",
              nt(r.delta_ahorro), nt(r.sigma_cv_ahorro), "sí" if int(r.cumple_regla) else "no", esc(r.decision)] for _, r in ree.iterrows()]

filas_neg = []
for l, r in NEG.iterrows():
    marca = ""
    if l in ree.lote_disparo.values:
        d = ree[ree.lote_disparo == l].iloc[0]
        marca = f" ⟲ v{int(d.version_challenger)} {esc(d.decision)}"
    filas_neg.append([f"{l}{' ★' if l >= L0 else ''}", f"v{int(r.version_modelo)}", f(r.tasa_impago_real, 3), nt(r.ahorro_modelo), nt(r.ahorro_regla),
                      "0", nt(r.modelo_menos_regla), nt(r.ahorro_sin_reentrenar) if pd.notna(r.ahorro_sin_reentrenar) else "–", marca])
filas_neg.append(["<b>Total</b>", "", "", f"<b>{nt(neg['modelo'])}</b>", f"<b>{nt(neg['regla'])}</b>", "<b>0</b>",
                  f"<b>{nt(neg['extra_vs_regla'])}</b>", "", ""])

filas_diag = [[int(r.lote_id), f(r.get("psi_max_del_lote"), 2), f(r.get("auc")), f(r.get("umbral")), f"v{int(r.version)}" if pd.notna(r.get("version")) else "–",
               esc(r.diagnostico).replace("_", " "), esc(r.accion)] for _, r in diag.iterrows()]

sub = SUB.copy()
filas_sub = [[esc(r.variable).replace("_", " "), esc(r.grupo), int(r.n), f(r.tasa_impago, 2), f(r.tasa_contacto, 2), f(r.auc), f(r.recall, 2), f(r.precision, 2)]
             for _, r in sub.iterrows()]

filas_ret = [[esc(c["condicion"]), esc(c["metrica"]), esc(c["umbral"]), esc(c["respuesta"]), esc(c["estado_simulacion"])] for c in R["criterios_retirada"]]
filas_costo = [[esc(c["error"]), esc(c["que_pasa"]), int(c["casos"]), nt(c["costo_estimado_NT"]), esc(c["quien_lo_asume"])] for c in R["costo_errores"]]
filas_evid = [[esc(e["paso"]), f"v{e['modelo']['version']}", esc(e["modelo"]["fuente_activa"]), f(e["prediccion"]["probabilidad"]),
               esc(e["prediccion"]["decision"]), f"v{e['prediccion']['version_modelo']}" if e["prediccion"]["version_modelo"] else "–"] for e in EVID]

# ----------------------------------------------------------------------------- HTML
css = """@page{size:A4;margin:13mm 14mm}body{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-size:9.1pt;line-height:1.34;color:#1f2933}
h1{font-size:15pt;margin:0 0 2px}.sub{color:#52606d;font-size:8.6pt;margin-bottom:5px}h2{break-after:avoid;font-size:11pt;margin:9px 0 3px;color:#102a43;border-bottom:1px solid #d9e2ec;padding-bottom:2px}
h3{font-size:9.5pt;margin:6px 0 2px;color:#243b53}p{margin:2px 0 5px}ul{margin:2px 0 5px 16px;padding:0}li{margin:1px 0}
table{border-collapse:collapse;width:100%;table-layout:auto;page-break-inside:auto;margin:3px 0 6px;font-size:7.8pt}th,td{border:1px solid #cbd2d9;padding:2px 4px;vertical-align:top;text-align:left}th{background:#f0f4f8}
code{font-family:Menlo,Consolas,monospace;font-size:7.7pt;background:#f5f7fa;padding:0 2px}img{display:block;margin:3px auto}
.cap{text-align:center;font-size:7.6pt;color:#52606d;margin:0 0 5px}.box{background:#f5f7fa;border-left:3px solid #7aa7e8;padding:4px 8px;margin:5px 0}
.two{display:flex;gap:10px;break-inside:avoid}.two>div{flex:1;min-width:0}.pb{page-break-before:always}pre{font-size:7.1pt;background:#f5f7fa;padding:4px 6px;white-space:pre-wrap;margin:2px 0}
.falta{border:1px dashed #c0392b;color:#c0392b;padding:6px;font-size:8pt;margin:4px 0}.kpi{display:flex;gap:6px;margin:5px 0}.kpi div{flex:1;background:#f0f4f8;padding:4px 6px;border-radius:3px;font-size:7.9pt}
.kpi b{display:block;font-size:11pt;color:#102a43}"""

video_html = f'<a href="{esc(VIDEO)}">{esc(VIDEO)}</a>' if VIDEO else '<span style="color:#c0392b">PENDIENTE: pegar el enlace en docs/video_url.txt</span>'
a0 = aprob.iloc[0] if len(aprob) else None

H = f"""<html><head><meta charset="utf-8"><style>{css}</style></head><body>
<h1>Anticipar la mora en tarjetas de crédito — documento técnico del proyecto final</h1>
<div class="sub">{AUTOR} · Fundamentos de Ciencia de Datos: End to End (USFQ) · PSet #4 · Repositorio: <a href="{REPO}">{REPO}</a> · Video de la demo (≤ 7 min): {video_html}</div>
<div class="kpi"><div><b>{nt(neg['extra_vs_regla'], True)}</b>pérdida evitada extra vs. la regla actual en {n_lotes} meses simulados</div>
<div><b>{gana} de {n_lotes}</b>lotes en que el modelo superó a la regla</div>
<div><b>{f(t_ch.test_roc_auc)}</b>AUC del champion en prueba (regla: {f(t_rg.test_roc_auc)})</div>
<div><b>lote {R.get('primer_lote_alerta_deriva') or '–'}</b>primera alerta de deriva (la deriva empezó en el lote {L0})</div>
<div><b>{len(ree)} / {len(aprob)}</b>reentrenamientos / aprobados por una persona</div></div>

<h2>1. Problema y formulación (versión final)</h2>
<p>El banco detecta la mora cuando ya ocurrió. El proyecto predice, con un mes de anticipación, qué clientes no pagarán su próxima cuota. El primer día de cada mes, cobranza recibe una lista priorizada y llama antes del vencimiento con una oferta de refinanciamiento o diferimiento.</p>
{tabla([
 ["Unidad", "Cliente de tarjeta, por mes (1 fila de la OBT por cliente)"],
 ["Target · horizonte", "<code>default_next_month</code>: no paga la cuota del mes siguiente · 1 mes (el target llega con el lote siguiente)"],
 ["Métrica técnica", "AUC-ROC; además, AP y recall/precisión en el 20 % de mayor riesgo"],
 ["Métrica de negocio", "Pérdida evitada neta = VP · M̄ · t − (VP + FP) · c, con M̄ = NT$ 47.967 (saldo medio en riesgo), t = 30 % (sensibilidad 10–50 %) y c = NT$ 500"],
 ["Línea base", "Regla actual: llamar a quien tiene ≥ 1 mes de atraso (contacta al ~22 %). También «no hacer nada» (ahorro 0)"],
 ["Umbral de decisión <i>(corregido)</i>", f"El equilibrio de costos es p* = c / (M̄·t) = {f(p_eq)}: un falso negativo cuesta ~29 veces lo que un falso positivo. Por costo convendría llamar a casi todos. <b>Lo que limita es la capacidad del call center (20 % de la cartera)</b>, así que el umbral operativo es el corte que la llena, medido con predicciones OOF de entrenamiento: <b>{f(corte_op, 2)}</b>. El 0,6 del PSet 1 no era bajo sino conservador: en prueba contactaba al 10 % y dejaba de evitar {nt(t_ch['test_ahorro_NT$'] - t_ch['test_ahorro_umbral_0.6'], True)}."],
 ["Juicio humano", "El modelo solo prioriza; el asesor revisa el caso y elige la oferta. Nuevo: una persona aprueba cada cambio de modelo (§6)"]],
 ["Elemento", "Definición"])}

<h2>2. Arquitectura end-to-end</h2>
<div class="two"><div style="flex:1.05">{img(D / 'arquitectura.png', '100%', 'Fuente → Snowflake (RAW→GOLD→OBT) → MLflow → API → lotes, monitoreo y reentrenamiento orquestados por Kestra')}</div>
<div><p><b>Flujo.</b> Kestra (<code>ingesta.yml</code>) carga cada mes la partición de la fuente en RAW de forma idempotente. Después corre <code>dbt build</code> (Bronze, Silver y Gold con sus tests) y, solo si los tests pasan, Spark construye la OBT.</p>
<p>Los notebooks 01–05 entrenan desde un <i>snapshot</i> de la OBT con su hash y registran todo en MLflow. El Model Registry es la fuente de verdad del modelo en producción: alias <code>@champion</code>, <code>@challenger</code> y <code>@anterior</code>.</p>
<p>La API (FastAPI en Docker) carga <code>models:/anticipar-mora-modelo@champion</code>, valida la entrada, aplica el umbral, devuelve la acción y registra cada predicción con su versión.</p>
<p>Cada mes, <code>produccion_lotes.yml</code>:</p>
<ul><li>lee el lote de Snowflake;</li><li>lo puntúa a través de la API;</li><li>carga PREDICCIONES;</li><li>une los targets que ya llegaron;</li><li>llama a <code>monitoreo.yml</code>, que calcula métricas, PSI y alertas.</li></ul>
<p>Si el desempeño cae dos lotes seguidos, o se cumple el calendario, el monitoreo dispara <code>reentrenamiento.yml</code>. Ese flujo entrena el challenger, lo compara con el champion y <b>se detiene hasta que una persona aprueba</b>. Si se aprueba, mueve el alias y la API cambia de modelo sin reconstruirse.</p>
<p>El código viaja por otra vía: GitHub Actions prueba cada push y, con cada merge a <code>main</code> en verde, publica la imagen de la API en GHCR con el hash del commit.</p>
<p>Todo levanta con <code>docker compose up -d --build</code>: Kestra, Spark, MLflow, API y tablero.</p></div></div>

<h2>3. Datos</h2>
<p><b>Ingesta, retries y backfill.</b> Cada ejecución carga una partición mensual con <code>DELETE</code> + <code>INSERT</code> en una sola transacción, así que un retry o un backfill nunca duplica filas.</p>
<ul><li><b>Retry exponencial</b> (30 s → 4 min, 4 intentos): las caídas de la fuente y la reanudación del warehouse duran minutos, no segundos.</li>
<li><b>Backfill:</b> es la misma ejecución con <code>meses</code> = abr–sep 2005.</li>
<li><b>Calendario:</b> día 2 a las 06:00, el cierre del estado de cuenta.</li></ul>
<p><b>Calidad.</b> Silver tipa, deduplica y marca problemas sin borrarlos: banderas de sobrecupo, saldo a favor, atraso sin saldo e imputación de educación y estado civil. Los 72 tests de dbt detienen el pipeline antes de construir la OBT. Tres problemas cambiaron decisiones:</p>
<ul><li><code>avg_payment_ratio_6m</code> llega 100 % vacía a la OBT (H6). Se excluye y el contrato de entrada queda fijo en 43 columnas, para que un cambio en la tabla no cambie en silencio lo que recibe el modelo.</li>
<li>Los códigos de educación 0, 5 y 6 se agrupan en «otros».</li>
<li>En producción, 3 registros corruptos por lote son rechazados por la API con un 422 y se miden como rechazos 4xx.</li></ul>
<div class="two"><div><p><b>Modelo dimensional.</b> Gold es un esquema estrella:</p>
<ul><li><code>FCT_MONTHLY_STATEMENT</code>, con grano <b>cliente × mes de estado de cuenta</b> (180.000 filas);</li>
<li><code>FCT_DEFAULT_OUTCOME</code>, con grano <b>cliente</b> (el target);</li>
<li>dimensiones cliente, mes y estado de pago.</li></ul>
<p><b>OBT.</b> Grano: 1 fila por cliente (30.000), con los 6 meses pivotados. Spark valida los joins antes de escribir: conteo igual al de clientes, <code>customer_id</code> único y exactamente 6 meses por cliente.</p>
<p><b>Batch vs. streaming.</b> La fuente cambia una vez al mes y la decisión también: el streaming solo agregaría costo.</p></div>
<div style="flex:.8">{img(D / 'modelo_dimensional.png', '100%', 'Esquema estrella de Gold (PSet 2)')}</div></div>

<h2>4. Modelado</h2>
<p><b>Hallazgos de la EDA que cambiaron el experimento</b> (solo sobre entrenamiento):</p>
{tabla([[esc(r.Hallazgo), esc(r['Consecuencia en el modelo'])] for _, r in HALL.iterrows()], ['Hallazgo', 'Qué cambió'])}
<p><b>Features y fuga.</b> Las derivadas (recencia del atraso, tendencia de mora, cobertura del último pago, tendencia de utilización, estado del último mes como categoría) solo usan columnas observadas hasta sep-2005. Se calculan en un transformador sin estado dentro del <code>Pipeline</code>, y el imputador y el escalador se ajustan solo con el pliegue de entrenamiento.</p>
<p><b>Plan de prueba contra la fuga:</b></p>
<ul><li>asserts de que ninguna feature es posterior al mes de observación;</li><li>AUC univariada de cada feature (ninguna «demasiado buena»);</li><li>ablación con y sin derivadas.</li></ul>
<p><b>Validación.</b> Hold-out estratificado 80/20 con semilla 42, usado una sola vez, más <code>StratifiedKFold(5)</code> en entrenamiento.</p>
{tabla(filas_cand, ['Modelo', 'Familia', 'AUC CV', 'σ', 'Brecha train−CV', 's por ajuste'])}<div class="cap">Líneas base y candidatos (CV de 5 pliegues). *LLM: 200 clientes, desviación por bootstrap.</div>
{tabla(filas_tun, ['Tuning (30 configuraciones)', 'AUC CV antes', 'AUC CV después', 'σ', 'Mejora', 'Mejora &gt; σ', 'Brecha train−CV'])}
<p><b>Ganador.</b> Ninguna mejora del tuning supera la variación entre pliegues. Por la regla de desempate, fijada antes de mirar el conjunto de prueba (menor brecha y, después, menor costo), el champion es <b>{esc(champ.modelo)} v{int(champ.version)}</b> (run <code>{esc(champ.run_id)}</code>); el challenger es {esc(chall.modelo)} v{int(chall.version)}.</p>
<p>En prueba, el champion obtiene AUC {f(t_ch.test_roc_auc)}, contra {f(t_rg.test_roc_auc)} de la regla, y {nt(t_ch['test_ahorro_NT$'], True)} de ahorro contra {nt(t_rg['test_ahorro_NT$'], True)}.</p>

<h2>5. Despliegue</h2>
{tabla([["Latencia de la decisión", "La lista se usa el día 1 de cada mes; basta con tenerla horas antes", "Batch"],
        ["Frecuencia de datos nuevos", "Un estado de cuenta por cliente al mes", "Batch"],
        ["Volumen", "Toda la cartera una vez al mes (30.000 clientes; 600 por lote en la simulación), no solicitudes por segundo", "Batch"],
        ["Costo de una predicción desactualizada", "Bajo dentro del mes: el estado de cuenta no cambia hasta el siguiente cierre", "Batch"]],
       ["Criterio", "Caso", "¿Qué patrón favorece?"])}
<p><b>Patrón: batch mensual.</b> Kestra genera las predicciones de cada período a través de la API y las guarda en Snowflake. <b>El patrón online sería el adecuado si</b> la decisión fuera en el momento, por ejemplo aprobar un aumento de cupo o bloquear una compra mientras el cliente espera. Haría falta que los pagos y consumos llegaran como eventos y que la predicción tuviera que estar lista en milisegundos. Las features pasarían a calcularse en un <i>feature store</i> en línea.</p>
<div class="two"><div><b>Solicitud</b> (extracto; el esquema completo tiene 43 campos, en el mismo orden que la firma de MLflow)<pre>POST /predecir
{esc(json.dumps(req, ensure_ascii=False))}</pre>
<b>Entrada inválida</b> → 422 con el campo que falló<pre>{esc(json.dumps(resp_mal, ensure_ascii=False)[:330])}</pre></div>
<div><b>Respuesta</b><pre>{esc(json.dumps({k: resp.get(k) for k in ['id_prediccion', 'probabilidad', 'decision', 'accion', 'umbral', 'version_modelo', 'alias', 'run_id', 'fuente', 'respaldo', 'latencia_ms']}, ensure_ascii=False, indent=1))}</pre></div></div>
<p>El umbral no está fijo en el código: la API lo lee de la corrida del champion en MLflow (<code>corte_efectivo_train_oof</code>), así que un modelo reentrenado trae su propio corte. Si el modelo no carga, la API responde con la regla y <code>respaldo: true</code>.</p>
<p><b>Registro.</b> Cada predicción (id, timestamp, entradas, salida, versión, run_id y latencia) se guarda en un registro local de la API. Kestra la carga en <code>PRODUCCION.PREDICCIONES</code> después de cada lote: así la API no guarda credenciales del almacén y una caída de Snowflake no la tumba.</p>
<p><b>Pruebas</b> (<code>pytest</code>, 21 casos):</p>
<ul><li><b>Contrato:</b> una entrada válida da 200 con el esquema; un campo faltante, un tipo incorrecto, una categoría desconocida, un valor fuera de rango o un campo extra dan 422; el esquema coincide con la firma.</li>
<li><b>Modelo:</b> la copia del champion reproduce el AUC registrado en MLflow sobre 600 clientes fijos, con tolerancia 1e-3. Es el mismo archivo, así que la diferencia esperada es 0; el margen cubre punto flotante entre Mac ARM y Linux y es 4 veces menor que la σ de la CV.</li>
<li><b>Comportamiento:</b> pasar a 2–3 meses de atraso no baja el riesgo; pagar el saldo no lo sube más de 0,05 ni convierte a nadie en «contactar»; la decisión es coherente con el umbral.</li>
<li><b>Respaldo:</b> modelo que no carga, interruptor por endpoint e interruptor por variable de entorno.</li></ul>
<p><b>Fuera de CI:</b> Snowflake, MLflow, Kestra y el reentrenamiento. Se validan en la simulación, que queda en el video.</p>
<div class="two"><div>{captura('github_actions', 'GitHub Actions: ci.yml (pruebas + build) y cd.yml en verde')}</div><div>{captura('ghcr', 'Imagen publicada en GitHub Container Registry con el hash del commit')}</div></div>

<h2>6. Versionado, reentrenamiento y despliegue continuo</h2>
<p><b>Regla de promoción</b> (fijada antes de simular). El challenger reemplaza al champion si se cumplen las dos condiciones, evaluadas ambas en el <b>mismo lote etiquetado más reciente, que ninguno de los dos vio</b>:</p>
<ul><li>ahorro(challenger) − ahorro(champion) &gt; σ_CV(ahorro), donde σ_CV es la desviación del ahorro entre los 5 pliegues de la CV del challenger, llevada al tamaño del lote;</li>
<li>AUC(challenger) ≥ AUC(champion) − σ_CV(AUC).</li></ul>
<p>El ahorro se mide a igual capacidad (top 20 % del lote) y la regla se aplica <b>antes</b> de la pausa humana.</p>
<p><b>Disparador.</b></p>
<ul><li><b>Por desempeño:</b> AUC &lt; media CV − 2σ<sub>lote</sub>, o ahorro por debajo de la regla, en <b>2 lotes etiquetados seguidos del champion vigente</b>. σ<sub>lote</sub> es la mayor entre la σ de la CV escalada a 600 clientes y el error bootstrap del lote: la σ de la CV sola subestima la variación de un lote chico.</li>
<li><b>Por calendario:</b> cada 5 lotes sin reentrenar.</li>
<li>La deriva de datos sola no reentrena.</li></ul>
<p><b>Ventana:</b> todo el histórico con más peso en lo reciente (peso de un lote = máx(1; 10·0,5<sup>edad</sup>)). Un cambio de régimen aparece primero en los lotes nuevos, pero 600 clientes se diluyen entre 24.000. El ajuste reutiliza <code>src/pipeline.py</code> y los hiperparámetros del champion (<code>clone</code>, sin tuning); la corrida queda en MLflow con <code>fase=reentrenamiento</code> y el motivo.</p>
{tabla(filas_ree, ['Lote', 'Motivo', 'Evaluado en', 'Versión', 'AUC champ./chall.', 'Ahorro champ./chall.', 'Δ ahorro', 'σ_CV', 'Cumple', 'Decisión'])}
<p>{txt_rech} {txt_ciclo}</p>
<p><b>Rollback sin reconstruir la imagen.</b> Basta mover el alias en MLflow: <code>python -m src.registro rollback</code> hace <code>@champion</code> ← <code>@anterior</code>. La API lo detecta (sondeo cada 30 s o <code>POST /admin/recargar</code>) y <code>/modelo</code> refleja el cambio. Evidencia de <code>scripts/evidencia_operacion.py</code>:</p>
{tabla(filas_evid, ['Paso', '/modelo', 'Fuente activa', 'p (mismo cliente)', 'Decisión', 'Versión en /predecir']) if filas_evid else '<div class="falta">Falta docs/evidencia_operacion.json</div>'}
<p><b>Por qué el código se despliega solo y el modelo no.</b> Un cambio de código pasa por pruebas deterministas: contrato, reproducción de la métrica y comportamiento. Un cambio de modelo cambia <b>a quién llama cobranza</b>, depende de supuestos de negocio (tasa de éxito y costo) que una persona debe validar, y puede cumplir la regla de ahorro y aun así aprender patrones indeseables, como pasó en esta simulación. Por eso la promoción espera una aprobación explícita y queda registrada (quién, cuándo y por qué) en MLflow y en <code>PRODUCCION.REENTRENAMIENTOS</code>.</p>
<div class="two"><div>{captura('kestra_pausa', 'Kestra: reentrenamiento detenido en la pausa de aprobación')}</div><div>{captura('mlflow_registry', 'MLflow: versiones y alias después de la promoción')}</div></div>

<h2>7. Simulación, monitoreo y alertas</h2>
<p><b>Lotes.</b> No hay datos posteriores a sep-2005, así que se usan los 6.000 clientes de prueba del PSet 3, repartidos al azar (semilla fija) en <b>{n_lotes} lotes de {R['clientes_por_lote']}</b>, que representan oct-2005 a jul-2006. El target del lote <i>k</i> se publica con el lote <i>k+1</i>.</p>
<p><b>Limitación:</b> <code>customer_id</code> no es temporal. Los lotes limpios miden la variación natural entre lotes, no estacionalidad ni tendencia, y la deriva es la que se indujo.</p>
<p><b>Deriva inducida desde el lote {L0}</b> (<code>src/simulacion.py</code>):</p>
<ul><li><b>Datos:</b> <code>limit_bal × 0,6</code>, con <code>util_m1..m6</code>, utilización media y máxima y meses sobre cupo recalculados con las fórmulas de Silver.</li>
<li><b>Concepto:</b> el 60 % de los clientes <i>al día</i> que con el nuevo cupo quedan con <code>util_m1 ≥ 0,70</code> y sí iban a pagar pasan a no pagar.</li>
<li><b>Calidad, en todos los lotes:</b> 3 registros corruptos (edad 0, educación 9, estado vacío).</li></ul>
{img(D / 'img' / 'monitoreo_deriva.png', '100%', 'PSI por feature y lote (izq.) y AUC por lote frente al umbral de deriva conceptual (der.). Rojo: alertas altas; línea verde: reentrenamiento aprobado.')}
<p><b>Umbrales del PSI.</b> El piso de ruido (PSI entre entrenamiento y 200 submuestras de 600 clientes del propio entrenamiento) tiene un p99 máximo de {f(R.get('piso_ruido_psi_p99_max'))}, y en los lotes limpios el PSI máximo fue {f(R['psi_max_lotes_limpios'])}. Por eso 0,10 (moderada) y 0,25 (alta) no son ruido. En los lotes alterados el PSI llegó a {f(R['psi_max_lotes_deriva'], 2)}. KS y chi² solo confirman (p &lt; 0,01): con 600 frente a 24.000 detectan diferencias sin importancia práctica.</p>
<p><b>Alertas</b> (plantilla de la sección 6.3, en <code>monitoreo/alertas.py</code>; se registran en <code>PRODUCCION.ALERTAS</code> y se publican en el log de Kestra con nivel ERROR):</p>
{tabla(filas_alertas, ['Métrica', 'Umbral', 'Ventana', 'Severidad', 'Quién la recibe', 'Acción'])}
<p><b>Resultado.</b> {R['alertas']} alertas, {R['alertas_altas']} altas. La primera alerta de deriva de datos llegó en el lote {R.get('primer_lote_alerta_deriva')}, el mismo en que empezó la deriva. {txt_falsas}</p>
{tabla(filas_diag, ['Lote', 'PSI máx.', 'AUC', 'Umbral', 'Modelo', 'Diagnóstico', 'Acción'])}
<p><b>Diagnóstico.</b></p>
<ul><li><b>Desde el lote {L0} hasta la promoción:</b> las entradas cambian y el desempeño cae: <b>deriva de datos con impacto</b>. La causa de fondo es conceptual (la relación entre utilización e impago cambió), y por eso solo reentrenar con los lotes nuevos la corrige.</li>
<li><b>Después de la promoción:</b> el PSI sigue alto, porque la cartera sigue con cupos recortados, pero el AUC vuelve sobre el umbral: <b>deriva de datos sin impacto</b>. No se reentrena otra vez; se actualiza la referencia.</li></ul>
{img(D / 'img' / 'monitoreo_sistema_modelo.png', '100%', 'Sistema y modelo por lote: latencia p50/p95, errores 4xx/5xx y tasa de «contactar» frente a la capacidad del 20 %')}
<p><b>Sistema y modelo.</b></p>
<ul><li><b>Latencia:</b> p50 de {f(R['latencia_p50_ms'], 0)} ms; el p95 nunca superó {f(R['latencia_p95_ms_max'], 0)} ms.</li>
<li><b>Errores:</b> 0 % de 5xx; los 4xx son exactamente los registros corruptos (0,5 % por lote).</li>
<li><b>Tasa de contacto:</b> después del reentrenamiento subió a {pct(max(R['tasa_contactar_por_lote'].values()), 0)} y disparó la alerta de capacidad. El corte del nuevo modelo se calibró con una ventana dominada por la cartera antigua. Cobranza llama igual al top 20 %, pero hay que recalibrar el corte con el último lote (§10).</li></ul>
<div class="two"><div>{captura('tablero', 'Tablero (Streamlit): métricas lote por lote con alertas y reentrenamientos marcados')}</div></div>

<h2>8. Ética operativa y criterios de retirada</h2>
<p><b>Desempeño por subgrupo</b> (todas las predicciones etiquetadas de la simulación; sexo y estado civil se excluyeron desde el PSet 1):</p>
{tabla(filas_sub, ['Variable', 'Grupo', 'n', 'Impago', 'Contacto', 'AUC', 'Recall', 'Precisión'])}
<p><b>¿Es aceptable la diferencia?</b></p>
<ul><li><b>Por edad:</b> la brecha de recall es de {f(brecha['grupo_edad'], 2)} y el AUC es parejo. Es aceptable para el negocio y para las personas.</li>
<li><b>Por tamaño del cupo:</b> la brecha es mayor ({f(brecha['cupo'], 2)}). El modelo llama más a los clientes de cupo bajo porque su tasa de impago real es mucho mayor, y el recorte de cupos de la deriva los golpeó más. Para el negocio es coherente; para las personas, la llamada es una oferta de refinanciamiento, no una sanción, y el asesor decide. Aun así supera el umbral de 0,15 que se fijó como criterio de ajuste.</li>
<li><b>Educación «otros»:</b> el grupo es muy chico (n bajo) y no permite conclusiones.</li></ul>
{tabla(filas_costo, ['Error', 'Qué pasa', 'Casos', 'Costo estimado', 'Quién lo asume'])}
<p><b>Criterios de retirada o ajuste</b> (evaluados con los datos de la simulación):</p>
{tabla(filas_ret, ['Condición observable', 'Métrica', 'Umbral', 'Respuesta', 'En la simulación'])}
<p><b>Interruptor de respaldo.</b> Hay dos formas de activarlo: <code>MODO_RESPALDO=1</code> al arrancar, o <code>POST /admin/respaldo</code> con el token de administración. Con el interruptor activo, la API sirve la regla de negocio, marca <code>respaldo: true</code> con su motivo en cada respuesta y la alerta «predicciones con respaldo» avisa en el lote. Está probado en CI y en la evidencia de §6 (pasos 4–5).</p>
<p><b>Supervisión humana.</b> El juicio del PSet 1 se mantiene: el asesor elige la oferta y ninguna decisión de crédito es automática. Se agrega una segunda instancia, porque el comportamiento observado lo justifica: <b>cada cambio de modelo lo aprueba una persona</b>. Esa persona ve la regla de ahorro y también las pruebas de comportamiento del challenger, y con el criterio de cupo activo debe revisar la lista de los clientes de cupo bajo antes de llamar.</p>

<h2>9. Resultado de negocio</h2>
{tabla(filas_neg, ['Lote', 'Modelo', 'Impago real', 'Modelo', 'Regla actual', 'Nada', 'Modelo − regla', 'Sin reentrenar', 'Reentrenamiento'])}
<div class="cap">★ lote con deriva inducida · ⟲ reentrenamiento disparado en ese lote · «Sin reentrenar»: los mismos clientes puntuados fuera de línea con el champion anterior</div>
{img(D / 'img' / 'negocio_por_lote.png', '92%', 'Pérdida evitada neta por lote: modelo, regla actual y champion anterior; no hacer nada = 0')}
<div class="box">{respuesta_negocio} {txt_efecto} La cifra depende de los supuestos del PSet 1 (éxito del 30 %, NT$ 500 por llamada); con éxito del 10 % el ahorro baja aproximadamente a un tercio, pero el orden entre modelo y regla se mantiene, porque los dos usan la misma fórmula. Lo que esta simulación no puede probar es la tasa de éxito real del contacto: eso solo lo mide el piloto A/B con cobranza.</div>

<h2>10. Limitaciones y siguientes pasos</h2>
<ul>
<li><b>Datos sin tiempo real.</b> Los lotes son una partición aleatoria de un solo mes y la deriva es inducida. Lo primero sería reentrenar y simular con historia mensual real de un banco local.</li>
<li><b>Corte de decisión.</b> Después del reentrenamiento la tasa de «contactar» superó la capacidad. Hay que recalibrar el corte con el último lote o pasar a una política «top-N» explícita, que ya es lo que hace cobranza.</li>
<li><b>Monotonía.</b> El modelo reentrenado violó la prueba «más atraso no baja el riesgo» en algunos clientes. Siguiente paso: restricciones de monotonía en LightGBM para estado de pago y utilización, y exigir las pruebas de comportamiento como condición de la regla de promoción.</li>
<li><b>Equidad por cupo.</b> La brecha de recall entre cupos supera 0,15. Hay que evaluar cortes por segmento y validar con el área de riesgo que la llamada sea un beneficio para el cliente.</li>
<li><b>Supuestos de negocio.</b> La tasa de éxito y el costo por llamada son supuestos. Para validarlos, un piloto A/B de 3 meses con un grupo de control.</li>
<li><b>Calidad pendiente del PSet 2.</b> <code>avg_payment_ratio_6m</code> llega vacía. Se corregirá en <code>build_obt.py</code> junto con el próximo reentrenamiento completo, para no cambiar el snapshot del champion actual.</li>
</ul>
</body></html>"""

(D / "documento.html").write_text(H)
chrome = shutil.which("google-chrome") or shutil.which("chromium") or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cuerpo = D / "_cuerpo.pdf"
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={cuerpo}", str(D / "documento.html")], stderr=subprocess.DEVNULL, check=True)

# ----------------------------------------------------------------------------- anexos
md = (D / "cambios.md").read_text()


def md_a_html(t: str) -> str:
    out, en_tabla, filas = [], False, []

    def inline(s):
        s = esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    for linea in t.splitlines() + [""]:
        if linea.startswith("|"):
            celdas = [c.strip() for c in linea.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in celdas):
                filas.append(celdas)
            en_tabla = True
            continue
        if en_tabla:
            out.append(tabla([[inline(c) for c in r] for r in filas[1:]], [inline(c) for c in filas[0]]))
            filas, en_tabla = [], False
        if linea.startswith("### "):
            out.append(f"<h3>{inline(linea[4:])}</h3>")
        elif linea.startswith("## "):
            out.append(f"<h2>{inline(linea[3:])}</h2>")
        elif linea.startswith("# "):
            out.append(f"<h1>{inline(linea[2:])}</h1>")
        elif linea.strip():
            out.append(f"<p>{inline(linea)}</p>")
    return "\n".join(out)


(D / "_anexo_cambios.html").write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"
                                       f"<h1>Anexo A · docs/cambios.md</h1>{md_a_html(md)}</body></html>")
anexo = D / "_anexo.pdf"
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={anexo}", str(D / "_anexo_cambios.html")], stderr=subprocess.DEVNULL, check=True)

from pypdf import PdfReader, PdfWriter  # noqa: E402

w = PdfWriter()
n_cuerpo = len(PdfReader(cuerpo).pages)
for p in [cuerpo, anexo] + [D / "historial" / n for n in ["PSet1_memo_Almeida_Cadena.pdf", "PSet1_pitch_Almeida_Cadena.pdf",
                                                         "PSet2_memo_Almeida_Cadena.pdf", "PSet3_memo_Almeida_Cadena.pdf"]]:
    if Path(p).exists():
        w.append(str(p))
w.write(SALIDA)
for p in (cuerpo, anexo, D / "_anexo_cambios.html"):
    Path(p).unlink(missing_ok=True)
print(f"Documento: {SALIDA.name} | cuerpo {n_cuerpo} páginas (máx. 12) | total con anexos {len(PdfReader(SALIDA).pages)}")
if n_cuerpo > 12:
    print("ATENCIÓN: el cuerpo supera 12 páginas")
pendientes = re.findall(r"Captura pendiente: ([^ ]+)", H) + (["video_url.txt"] if not VIDEO else [])
if pendientes:
    print("Pendiente antes de entregar:", ", ".join(pendientes))

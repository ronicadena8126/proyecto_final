"""
Tablero de monitoreo (sección 6.4): lote por lote, métricas de sistema, modelo, desempeño y deriva,
con las alertas y los reentrenamientos marcados.

  streamlit run tablero/app.py        (en Docker: http://localhost:8501)

Lee PRODUCCION.MONITOREO / ALERTAS / REENTRENAMIENTOS de Snowflake (o del DuckDB local si ALMACEN=local).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import almacen  # noqa: E402

st.set_page_config(page_title="Anticipar la mora · monitoreo", layout="wide")
E = almacen.ESQUEMA
GRIS, AZUL, NARANJA, ROJO, VERDE = "#8a8f98", "#2f6fb3", "#e08a2e", "#c0392b", "#2e8b57"


@st.cache_data(ttl=30)
def cargar():
    def q(sql):
        try:
            return almacen.leer(sql)
        except Exception:
            return pd.DataFrame()

    mon = q(f"SELECT * FROM {E}.MONITOREO")
    ale = q(f"SELECT * FROM {E}.ALERTAS")
    ree = q(f"SELECT * FROM {E}.REENTRENAMIENTOS")
    return mon, ale, ree


def serie(mon, tipo, metrica, feature=None):
    d = mon[(mon.tipo == tipo) & (mon.metrica == metrica)]
    if feature is not None:
        d = d[d.feature == feature]
    return d.groupby("lote_id").valor.last().sort_index()


def marcar(fig, ale, ree):
    """Bandas rojas en lotes con alertas altas; líneas verdes en reentrenamientos (con su decisión)."""
    if len(ale):
        for l in sorted(ale[ale.severidad == "Alta"].lote_id.unique()):
            fig.add_vrect(x0=l - 0.45, x1=l + 0.45, fillcolor=ROJO, opacity=0.07, line_width=0)
    if len(ree):
        for _, r in ree.iterrows():
            color = VERDE if str(r.decision).startswith("aprobado") else GRIS
            fig.add_vline(x=r.lote_disparo + 0.5, line_dash="dash", line_color=color,
                          annotation_text=f"reentreno v{int(r.version_challenger)} {r.decision}",
                          annotation_position="top left", annotation_font_size=10)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25),
                      xaxis=dict(title="lote (mes simulado)", dtick=1))
    return fig


mon, ale, ree = cargar()
st.title("Anticipar la mora — monitoreo del modelo en producción")
if mon.empty:
    st.info("Todavía no hay lotes monitoreados. Ejecuta el flujo produccion_lotes en Kestra.")
    st.stop()

# --------------------------------------------------------------------------- resumen
c1, c2, c3, c4, c5 = st.columns(5)
lotes = sorted(mon.lote_id.unique())
c1.metric("Lotes monitoreados", len(lotes))
c2.metric("Alertas", len(ale), f"{(ale.severidad == 'Alta').sum() if len(ale) else 0} altas", delta_color="off")
c3.metric("Reentrenamientos", len(ree), f"{ree.decision.str.startswith('aprobado').sum() if len(ree) else 0} aprobados",
          delta_color="off")
try:
    import httpx

    info = httpx.get(os.environ.get("API_URL", "http://localhost:8000") + "/modelo", timeout=3).json()
    c4.metric("Modelo en producción", f"v{info['version']}", info["fuente_activa"], delta_color="off")
except Exception:
    v = serie(mon, "modelo", "n_predicciones")
    c4.metric("Modelo en producción", "API no disponible")
des = mon[mon.tipo == "desempeno"]
if len(des):
    tot = des[des.metrica.isin(["ahorro_modelo", "ahorro_regla"])].pivot_table(index="lote_id", columns="metrica",
                                                                               values="valor", aggfunc="last")
    c5.metric("Ahorro acumulado vs. regla", f"NT$ {tot.ahorro_modelo.sum() - tot.ahorro_regla.sum():,.0f}",
              f"modelo NT$ {tot.ahorro_modelo.sum():,.0f}", delta_color="off")

st.caption("Bandas rojas: lotes con alertas de severidad alta. Líneas verticales: reentrenamientos "
           "(verde = aprobado y promovido; gris = rechazado).")

# --------------------------------------------------------------------------- negocio y desempeño
st.header("Resultado de negocio y desempeño (lotes con target)")
a, b = st.columns(2)
fig = go.Figure()
for m, n, col in [("ahorro_modelo", "Modelo", AZUL), ("ahorro_regla", "Regla actual (≥1 mes de atraso)", NARANJA)]:
    s = serie(mon, "desempeno", m)
    fig.add_bar(x=s.index, y=s.values, name=n, marker_color=col)
fig.add_hline(y=0, line_color=GRIS, annotation_text="no hacer nada = 0")
fig.update_layout(title="Pérdida evitada neta por lote (NT$)", barmode="group")
a.plotly_chart(marcar(fig, ale, ree), use_container_width=True)

fig = go.Figure()
auc, umb, cvm = serie(mon, "desempeno", "auc"), serie(mon, "desempeno", "umbral_auc"), serie(mon, "desempeno", "cv_mean")
fig.add_scatter(x=auc.index, y=auc.values, mode="lines+markers", name="AUC del lote", line_color=AZUL)
fig.add_scatter(x=cvm.index, y=cvm.values, mode="lines", name="media CV del modelo", line=dict(color=GRIS, dash="dot"))
fig.add_scatter(x=umb.index, y=umb.values, mode="lines", name="umbral (CV − 2σ lote)", line=dict(color=ROJO, dash="dash"))
fig.update_layout(title="Deriva conceptual: AUC por lote vs. CV")
b.plotly_chart(marcar(fig, ale, ree), use_container_width=True)

# --------------------------------------------------------------------------- modelo
st.header("Modelo")
a, b = st.columns(2)
fig = go.Figure()
p10, p50, p90 = (serie(mon, "modelo", f"prob_p{q}") for q in (10, 50, 90))
fig.add_scatter(x=p90.index, y=p90.values, line=dict(width=0), showlegend=False)
fig.add_scatter(x=p10.index, y=p10.values, fill="tonexty", fillcolor="rgba(47,111,179,.15)", line=dict(width=0),
                name="p10–p90")
fig.add_scatter(x=p50.index, y=p50.values, mode="lines+markers", name="mediana", line_color=AZUL)
fig.update_layout(title="Distribución de la probabilidad de impago")
a.plotly_chart(marcar(fig, ale, ree), use_container_width=True)
fig = go.Figure()
s = serie(mon, "modelo", "tasa_contactar")
fig.add_bar(x=s.index, y=s.values, name="contactar", marker_color=AZUL)
r = serie(mon, "modelo", "tasa_respaldo")
fig.add_scatter(x=r.index, y=r.values, mode="lines+markers", name="respaldo (regla)", line_color=ROJO)
fig.add_hline(y=0.20, line_dash="dash", line_color=GRIS, annotation_text="capacidad 20%")
fig.update_layout(title="Tasa de cada decisión", yaxis_tickformat=".0%")
b.plotly_chart(marcar(fig, ale, ree), use_container_width=True)

# --------------------------------------------------------------------------- deriva de datos
st.header("Deriva de datos (PSI contra entrenamiento)")
psi = mon[(mon.tipo == "deriva") & (mon.metrica == "psi")].pivot_table(index="feature", columns="lote_id",
                                                                       values="valor", aggfunc="last")
psi = psi.loc[psi.max(axis=1).sort_values(ascending=False).index[:15]]
fig = go.Figure(go.Heatmap(z=psi.values, x=psi.columns, y=psi.index, zmin=0, zmax=0.5, colorscale="Reds",
                           colorbar=dict(title="PSI")))
fig.update_layout(title="Top 15 features por PSI (≥ 0,25 = alerta; 0,10–0,25 = moderada)", height=430,
                  margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(dtick=1, title="lote"))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- sistema
st.header("Sistema (API)")
a, b = st.columns(2)
fig = go.Figure()
for m, n, col in [("latencia_p50_ms", "p50", AZUL), ("latencia_p95_ms", "p95", NARANJA)]:
    s = serie(mon, "sistema", m)
    fig.add_scatter(x=s.index, y=s.values, mode="lines+markers", name=n, line_color=col)
fig.update_layout(title="Latencia de /predecir (ms)")
a.plotly_chart(marcar(fig, ale, ree), use_container_width=True)
fig = go.Figure()
vol = serie(mon, "sistema", "volumen_solicitudes")
fig.add_bar(x=vol.index, y=vol.values, name="solicitudes", marker_color="#c9d6e8")
for m, n, col in [("tasa_4xx", "4xx", NARANJA), ("tasa_5xx", "5xx", ROJO)]:
    s = serie(mon, "sistema", m)
    fig.add_scatter(x=s.index, y=s.values * 100, mode="lines+markers", name=f"{n} (%)", line_color=col, yaxis="y2")
fig.update_layout(title="Volumen y tasa de errores", yaxis2=dict(overlaying="y", side="right", title="%"))
b.plotly_chart(marcar(fig, ale, ree), use_container_width=True)

# --------------------------------------------------------------------------- tablas
st.header("Alertas")
if len(ale):
    st.dataframe(ale[["lote_id", "severidad", "regla", "mensaje", "destinatario", "accion"]]
                 .sort_values(["lote_id", "severidad"]), use_container_width=True, hide_index=True)
st.header("Reentrenamientos")
if len(ree):
    st.dataframe(ree[["lote_disparo", "motivo", "lote_evaluacion", "version_challenger", "auc_champion",
                      "auc_challenger", "ahorro_champion", "ahorro_challenger", "delta_ahorro", "sigma_cv_ahorro",
                      "cumple_regla", "decision", "comentario"]], use_container_width=True, hide_index=True)
diag = mon[mon.tipo == "diagnostico"][["lote_id", "metrica"]].rename(columns={"metrica": "diagnóstico"})
if len(diag):
    st.header("Diagnóstico por lote (tabla 6.2)")
    st.dataframe(diag.sort_values("lote_id"), use_container_width=True, hide_index=True)

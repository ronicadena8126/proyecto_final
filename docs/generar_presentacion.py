"""
Presentación ejecutiva (11 diapositivas + anexos técnicos) para el patrocinador del PSet 1:

  python docs/generar_presentacion.py  ->  docs/PSet4_presentacion_Almeida_Cadena.pdf

Mismas fuentes de datos que el documento técnico (docs/resumen_simulacion.json y CSV del notebook 07).
Regla de estilo: ninguna métrica técnica aparece sin su consecuencia de negocio.
"""
from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent
R = json.loads((D / "resumen_simulacion.json").read_text())
NEG = pd.read_csv(D / "negocio_por_lote.csv", index_col=0)
REE = pd.read_csv(D / "reentrenamientos.csv").sort_values("lote_disparo")
SUB = pd.read_csv(D / "subgrupos.csv")
TEST = pd.read_csv(D / "pset3" / "resultados_test.csv").set_index("modelo")
REG = pd.read_csv(D / "pset3" / "registro.csv", index_col=0)
AUTOR = "Ronnie Almeida Cadena"


def f(x, d=1):
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def m(x, d=2):
    return f"NT$ {f(x / 1e6, d)} M"


def esc(s):
    return html.escape(str(s))


def img(p, estilo="width:100%"):
    p = Path(p)
    if not p.exists():
        return f"<div class='falta'>Falta {esc(p.name)}</div>"
    return f"<img style='{estilo}' src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'>"


neg = R["negocio"]
n = int(R["n_lotes"])
L0 = int(R["inicio_deriva"])
aprob = REE[REE.decision.astype(str).str.startswith("aprobado")]
rech = REE[REE.decision.astype(str) == "rechazado"]
champ = REG.loc["champion"]
t_ch, t_rg = TEST.loc[champ.modelo], TEST.loc["regla_negocio"]
brecha = R["subgrupos_brecha_recall"]
cupo = SUB[SUB.variable == "cupo"].set_index("grupo")
tasa_max = max(R["tasa_contactar_por_lote"].values())
a = aprob.iloc[0] if len(aprob) else None
extra_pct = neg["extra_vs_regla"] / neg["regla"]

# Gráfico simple de negocio (barras modelo vs regla por mes simulado)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

meses = ["oct", "nov", "dic", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep"][:len(NEG)]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
x = range(len(NEG))
ax.bar([i - .2 for i in x], NEG.ahorro_modelo / 1e3, .4, color="#2f6fb3", label="Con el modelo")
ax.bar([i + .2 for i in x], NEG.ahorro_regla / 1e3, .4, color="#e3a35a", label="Regla actual")
for i, (l, r) in enumerate(NEG.iterrows()):
    if l >= L0:
        ax.axvspan(i - .5, i + .5, color="#c0392b", alpha=.06, lw=0)
if a is not None:
    i = list(NEG.index).index(int(a.lote_disparo))
    ax.axvline(i + .5, color="#2e8b57", ls="--")
    ax.text(i + .55, ax.get_ylim()[1] * .95, "modelo actualizado\n(aprobado por una persona)", fontsize=8, color="#2e8b57", va="top")
ax.set_xticks(list(x)); ax.set_xticklabels(meses); ax.set_ylabel("pérdida evitada (miles NT$)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(list(NEG.index).index(L0) - .4, 30, "cambio en el mercado →", fontsize=8, color="#c0392b")
plt.tight_layout(); plt.savefig(D / "img" / "pres_negocio.png", dpi=170); plt.close()

css = """@page{size:13.333in 7.5in;margin:0}*{box-sizing:border-box}body{margin:0;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;color:#1f2933}
.s{width:13.333in;height:7.5in;padding:.55in .7in;page-break-after:always;position:relative;overflow:hidden}
.s h1{font-size:30pt;margin:0 0 .15in;color:#102a43;line-height:1.1}.s h2{font-size:15pt;margin:0 0 .3in;color:#52606d;font-weight:400}
.k{font-size:10pt;text-transform:uppercase;letter-spacing:.08em;color:#2f6fb3;margin-bottom:.12in;font-weight:600}
.cols{display:flex;gap:.35in}.cols>div{flex:1}.card{background:#f0f4f8;border-radius:10px;padding:.22in .25in}
.card b.big{display:block;font-size:30pt;color:#102a43;line-height:1.05;margin-bottom:.06in}.card p{font-size:12.5pt;margin:.04in 0;line-height:1.35}
p,li{font-size:14pt;line-height:1.4}ul{margin:.05in 0 0 .25in;padding:0}li{margin:.07in 0}
.pie{position:absolute;bottom:.28in;left:.7in;right:.7in;font-size:9pt;color:#7b8794;display:flex;justify-content:space-between}
table{border-collapse:collapse;width:100%;font-size:12pt}td,th{padding:.08in .1in;border-bottom:1px solid #d9e2ec;text-align:left;vertical-align:top}th{color:#52606d;font-weight:600}
.flujo{display:flex;align-items:stretch;gap:.12in;margin:.25in 0}.flujo div{flex:1;background:#f0f4f8;border-radius:10px;padding:.2in;font-size:12.5pt}
.flujo div b{display:block;font-size:14pt;color:#102a43;margin-bottom:.05in}.flecha{flex:0 0 .3in !important;background:none !important;font-size:24pt !important;color:#9aa5b1;display:flex;align-items:center;justify-content:center;padding:0 !important}
.portada{background:#102a43;color:#fff}.portada h1{color:#fff;font-size:40pt}.portada h2{color:#bcccdc}.verde{color:#2e8b57}.rojo{color:#c0392b}
.rec{border-left:6px solid #2e8b57;padding-left:.25in}.falta{border:1px dashed #c0392b;color:#c0392b;padding:.2in}"""


def pie(i):
    return f"<div class='pie'><span>Anticipar la mora · resultados del piloto simulado</span><span>{i}</span></div>"


S = []
S.append(f"""<div class='s portada'><div style='margin-top:1.4in'><div class='k' style='color:#9fb3c8'>Proyecto final · Ciencia de Datos End to End</div>
<h1>De cobrar tarde a prevenir a tiempo</h1><h2>Resultados de operar el modelo de mora durante {n} meses simulados y recomendación</h2>
<p style='color:#d9e2ec;margin-top:.6in'>{AUTOR} · USFQ · noviembre 2026</p></div></div>""")

S.append(f"""<div class='s'><div class='k'>El problema</div><h1>Hoy cobranza llama cuando el cliente ya dejó de pagar</h1>
<div class='cols'><div class='card'><b class='big'>22 %</b><p>de los clientes de tarjeta no paga su cuota el mes siguiente.</p></div>
<div class='card'><b class='big'>NT$ 47.967</b><p>es el saldo promedio en riesgo de cada cliente que cae en mora.</p></div>
<div class='card'><b class='big'>1 mes tarde</b><p>La regla actual solo llama a quien <b>ya</b> tiene un mes de atraso, cuando recuperar es más caro y el cliente ya tiene intereses de mora y una peor calificación.</p></div></div>
<p style='margin-top:.35in'><b>Para quién:</b> el área de cobranza (a quién llamar primero), el banco (menos provisiones y castigos) y el cliente (una oferta de refinanciamiento antes de caer en mora).</p>{pie(2)}</div>""")

S.append(f"""<div class='s'><div class='k'>Qué cambió en la decisión</div><h1>De «llamar a los atrasados» a «llamar primero a quien va a caer»</h1>
<div class='cols'><div class='card'><p style='font-weight:600;color:#52606d'>ANTES · regla actual</p><ul><li>Llama a todo cliente con ≥ 1 mes de atraso (≈ 22 % de la cartera).</li><li>Sin orden de prioridad.</li><li>No ve a quien está al día pero va a dejar de pagar.</li></ul></div>
<div class='card'><p style='font-weight:600;color:#2f6fb3'>AHORA · lista del modelo</p><ul><li>El día 1 de cada mes, cobranza recibe una lista ordenada por riesgo de impago.</li><li>Se llama al 20 % de mayor riesgo: lo que el call center puede atender.</li><li><b>El asesor sigue decidiendo</b> la oferta; nada es automático.</li></ul></div></div>
<p style='margin-top:.3in'><b>Por qué hasta el 20 % y no un umbral fijo:</b> una llamada cuesta NT$ 500, y un impago evitado vale cerca de NT$ 14.400 (30 % de éxito sobre el saldo en riesgo). Llamar conviene desde un 3,5 % de riesgo, así que <b>el límite real es la capacidad del equipo</b>, no el costo. El umbral de 0,6 del pitch inicial dejaba sin llamar a la mitad de la capacidad.</p>{pie(3)}</div>""")

S.append(f"""<div class='s'><div class='k'>Cuánto valor generó</div><h1>{m(neg['extra_vs_regla'])} más de pérdida evitada que la regla actual en {n} meses</h1>
<div class='cols'><div style='flex:1.7'>{img(D / 'img' / 'pres_negocio.png')}</div>
<div><div class='card' style='margin-bottom:.15in'><b class='big'>{m(neg['modelo'])}</b><p>pérdida evitada con el modelo, frente a {m(neg['regla'])} con la regla (<b>+{f(100 * extra_pct)} %</b>) y NT$ 0 sin hacer nada.</p></div>
<div class='card'><b class='big'>{neg['lotes_modelo_gana']} de {n}</b><p>meses en que el modelo superó a la regla, con el mismo equipo de cobranza.</p></div></div></div>
<p style='font-size:11pt;color:#52606d;margin-top:.12in'>Simulación con 6.000 clientes reales que el modelo nunca vio, repartidos en {n} meses. Supuestos del pitch: 30 % de éxito del contacto y NT$ 500 por llamada; con 10 % de éxito el valor baja a un tercio, pero el modelo sigue por encima de la regla.</p>{pie(4)}</div>""")

S.append(f"""<div class='s'><div class='k'>Cómo funciona</div><h1>Una cadena mensual automática, con una persona en los puntos que importan</h1>
<div class='flujo'><div><b>1 · Datos del cierre</b>Cada mes se cargan pagos, saldos y uso del cupo, con controles de calidad automáticos.</div><div class='flecha'>→</div>
<div><b>2 · Puntaje de riesgo</b>El modelo vigente da a cada cliente su probabilidad de no pagar y la acción sugerida.</div><div class='flecha'>→</div>
<div><b>3 · Lista para cobranza</b>Top 20 % por riesgo; <b>el asesor elige la oferta</b>.</div><div class='flecha'>→</div>
<div><b>4 · Vigilancia</b>Cada mes se compara lo predicho con lo que pasó; si el modelo pierde valor, se propone uno nuevo y <b>una persona aprueba</b> el cambio.</div></div>
<p>Si algo falla, un interruptor vuelve en segundos a la regla actual, sin tocar sistemas.</p><p style='font-size:11pt;color:#52606d'>Detalle técnico en el anexo B.</p>{pie(5)}</div>""")

if a is not None:
    S.append(f"""<div class='s'><div class='k'>Qué pasa cuando el mercado cambia</div><h1>El sistema detectó el cambio el mismo mes y lo corrigió con aprobación humana</h1>
<div class='cols'><div class='card'><p style='font-weight:600'>Mes {L0} · el banco recorta cupos y los clientes muy endeudados empiezan a fallar</p><p>La alerta de «cartera distinta» se encendió <b>ese mismo mes</b>.</p></div>
<div class='card'><p style='font-weight:600'>Meses {L0}–{int(a.lote_disparo) - 1} · el modelo pierde precisión</p><p>Dos meses seguidos acertando menos de lo esperado: el sistema entrenó un modelo actualizado y lo puso a prueba con clientes que ninguno de los dos modelos había visto. Habría evitado <b>{m(a.delta_ahorro, 3)}</b> más ese mes.</p></div>
<div class='card'><p style='font-weight:600' class='verde'>Mes {int(a.lote_disparo)} · una persona aprobó el cambio</p><p>Con el modelo actualizado, los meses siguientes evitaron <b>{m(neg['efecto_reentrenar'], 2)}</b> más que si no se hubiera actualizado.</p></div></div>
<p style='margin-top:.3in'>{"En el mes " + str(int(rech.iloc[0].lote_disparo)) + " también se probó una actualización de rutina que <b>no mejoraba</b> y se rechazó: el sistema no cambia de modelo por cambiar. " if len(rech) else ""}<b>La aprobación humana importa:</b> el modelo propuesto cumplía la meta de ahorro, pero en algunos clientes al día y muy endeudados los calificaba más riesgosos que a otros con dos meses de atraso. La persona que aprueba lo ve antes de decidir.</p>{pie(6)}</div>""")

S.append(f"""<div class='s'><div class='k'>Riesgos que quedan y cómo se vigilan</div><h1>Cuatro riesgos, cada uno con su señal y su responsable</h1>
<table><tr><th>Riesgo</th><th>Cómo lo vemos</th><th>Quién actúa</th><th>Qué hacemos</th></tr>
<tr><td>La lista crece más que la capacidad (llegó al <b>{f(100 * tasa_max, 0)} %</b> tras la actualización)</td><td>Alerta mensual si «contactar» pasa del 25 %</td><td>Gerencia de cobranza</td><td>Se llama al top 20 %; recalibrar el corte con el último mes</td></tr>
<tr><td>Se llama más a clientes de <b>cupo bajo</b>: llegamos al {f(100 * cupo.loc['cupo bajo', 'recall'], 0)} % de sus morosos, frente al {f(100 * cupo.loc['cupo alto', 'recall'], 0)} % en cupo alto</td><td>Tablero por segmento cada mes</td><td>Riesgo + cobranza</td><td>Revisar el corte por segmento; la llamada debe ser una oferta, no una presión</td></tr>
<tr><td>Un modelo nuevo aprende patrones sin sentido de negocio</td><td>Pruebas automáticas de comportamiento en cada propuesta</td><td>Quien aprueba el cambio</td><td>No se promueve sin revisarlas; agregar reglas de coherencia al entrenamiento</td></tr>
<tr><td>El 30 % de éxito del contacto es un supuesto</td><td>Solo un piloto lo mide</td><td>Patrocinador</td><td>Piloto A/B de 3 meses (diapositiva 10)</td></tr></table>{pie(7)}</div>""")

S.append(f"""<div class='s'><div class='k'>Si algo sale mal</div><h1>Podemos volver a la regla actual en segundos y sabemos cuándo hacerlo</h1>
<div class='cols'><div class='card'><p style='font-weight:600'>Interruptor de respaldo</p><p>Con un solo comando, el servicio deja de usar el modelo y aplica la regla actual, avisando en cada respuesta. Probado en la simulación y en las pruebas automáticas.</p></div>
<div class='card'><p style='font-weight:600'>Volver a la versión anterior</p><p>Cada versión del modelo queda guardada; volver a la anterior toma segundos y no requiere reinstalar nada.</p></div></div>
<table style='margin-top:.25in'><tr><th>Si pasa esto…</th><th>…hacemos esto</th></tr>
<tr><td>El modelo ahorra menos que la regla 2 meses seguidos, aun después de actualizarlo</td><td>Interruptor a la regla y revisión</td></tr>
<tr><td>El modelo acierta poco más que la regla (2 meses seguidos tras una actualización)</td><td>Retirar el modelo y rediseñar</td></tr>
<tr><td>Un segmento de clientes queda claramente peor atendido</td><td>Ajustar la lista de ese segmento; la decisión vuelve a ser 100 % del asesor</td></tr></table>{pie(8)}</div>""")

S.append(f"""<div class='s'><div class='k'>Recomendación</div><h1>Ajustar y escalar a un piloto controlado</h1>
<div class='rec'><p><b>Por qué no retirar:</b> el modelo evitó {m(neg['extra_vs_regla'])} más que la regla en {n} meses ({neg['lotes_modelo_gana']}/{n} meses mejor) y se recuperó solo de un cambio fuerte de mercado.</p>
<p><b>Por qué no escalar todavía a toda la cartera:</b> los datos son de otro país y otra época, el éxito del contacto es un supuesto, y hay dos ajustes pendientes: el tamaño de la lista tras actualizar y la equidad por cupo.</p></div>
<p style='margin-top:.25in'><b>Qué se necesita</b> (3 meses):</p><ul>
<li><b>Datos locales:</b> 12 meses de historial anonimizado de una cartera de tarjetas, para reentrenar.</li>
<li><b>Piloto A/B con cobranza:</b> un equipo usa la lista y otro sigue con la regla. Medimos la mora real a 3 meses y el éxito del contacto.</li>
<li><b>Dos ajustes técnicos</b> antes del piloto: tamaño fijo de la lista y reglas de coherencia en el entrenamiento.</li>
<li><b>Un responsable de negocio</b> que apruebe cada cambio de modelo (≈ 1 hora al mes).</li></ul>{pie(9)}</div>""")

S.append(f"""<div class='s'><div class='k'>En resumen</div><h1>Qué resolvimos, para quién y qué sigue</h1>
<table><tr><td style='width:28%'><b>Problema</b></td><td>La mora se detectaba un mes tarde; cobranza llamaba sin prioridad.</td></tr>
<tr><td><b>Decisión</b></td><td>Lista mensual del 20 % de mayor riesgo; el asesor elige la oferta.</td></tr>
<tr><td><b>Valor</b></td><td><b>{m(neg['extra_vs_regla'])}</b> más de pérdida evitada que la regla en {n} meses (+{f(100 * extra_pct)} %).</td></tr>
<tr><td><b>Riesgos vigilados</b></td><td>Cambios de mercado, tamaño de la lista y equidad por segmento, con alerta mensual, responsable y acción.</td></tr>
<tr><td><b>Recomendación</b></td><td><b>Ajustar y pasar a un piloto A/B de 3 meses</b> con datos locales.</td></tr></table>
<p style='margin-top:.4in;font-size:18pt;color:#102a43'><b>Pedimos:</b> datos locales, un equipo de cobranza para el piloto y un responsable de negocio que apruebe los cambios.</p>{pie(10)}</div>""")

# ------------------------------------------------------------------ anexos técnicos
S.append(f"""<div class='s'><div class='k'>Anexo A · Detalle técnico (cada cifra con su consecuencia)</div><h1 style='font-size:24pt'>Calidad del modelo y de la operación</h1>
<table><tr><th>Medida técnica</th><th>Valor</th><th>Qué significa para el negocio</th></tr>
<tr><td>AUC en clientes nunca vistos (modelo / regla)</td><td>{f(t_ch.test_roc_auc, 3)} / {f(t_rg.test_roc_auc, 3)}</td><td>Ordena mejor a los clientes por riesgo: con las mismas llamadas, llega a más morosos.</td></tr>
<tr><td>Morosos encontrados en el 20 % de mayor riesgo</td><td>{f(100 * t_ch.test_recall_top20, 0)} % de todos los morosos</td><td>Con la capacidad actual, {f(100 * t_ch.test_recall_top20, 0)} de cada 100 clientes que van a caer reciben una oferta antes.</td></tr>
<tr><td>Morosos detectados con las mismas llamadas que la regla</td><td>{int(t_ch.moras_igual_volumen_regla)} vs. {int(t_rg.moras_detectadas)}</td><td>Más clientes ayudados a tiempo sin contratar más asesores.</td></tr>
<tr><td>Tiempo de respuesta del servicio (p95)</td><td>&lt; {f(R['latencia_p95_ms_max'], 0)} ms</td><td>Toda la cartera se puntúa en minutos; la lista llega a tiempo el día 1.</td></tr>
<tr><td>Registros rechazados por datos inválidos</td><td>{R['rechazos_4xx']} de {R['predicciones'] + R['rechazos_4xx']}</td><td>Ningún cliente se califica con datos corruptos; el error se reporta al origen.</td></tr>
<tr><td>Detección del cambio de mercado</td><td>mes {R.get('primer_lote_alerta_deriva')}; 0 falsas alarmas de deriva</td><td>El equipo se entera el mismo mes, sin alarmas que lo distraigan.</td></tr>
<tr><td>Precisión mínima aceptada por mes</td><td>media de la validación − 2 desviaciones</td><td>Por debajo de ese nivel dos meses seguidos se propone actualizar el modelo.</td></tr></table>{pie(11)}</div>""")

S.append(f"""<div class='s'><div class='k'>Anexo B · Arquitectura técnica</div><div style='display:flex;gap:.3in'><div style='flex:1'>{img(D / 'arquitectura.png', 'height:6.1in')}</div>
<div style='flex:1;font-size:12pt'><p style='font-size:12pt'><b>Datos:</b> Kestra → Snowflake (RAW → BRONZE → SILVER → GOLD con dbt) → OBT con Spark.</p><p style='font-size:12pt'><b>Modelo:</b> experimentos y registro de versiones en MLflow; champion: {esc(champ.modelo)}.</p>
<p style='font-size:12pt'><b>Servicio:</b> API FastAPI en Docker; carga siempre la versión aprobada, valida entradas y guarda cada predicción con su versión.</p>
<p style='font-size:12pt'><b>Operación:</b> Kestra puntúa cada mes, monitorea (calidad, deriva, precisión, ahorro) y propone reentrenamientos que una persona aprueba; tablero en Streamlit.</p>
<p style='font-size:12pt'><b>Código:</b> pruebas automáticas en cada cambio y publicación de la imagen del servicio con GitHub Actions.</p></div></div>{pie(12)}</div>""")

h = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(S)}</body></html>"
(D / "presentacion.html").write_text(h)
chrome = shutil.which("google-chrome") or shutil.which("chromium") or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out = D / "PSet4_presentacion_Almeida_Cadena.pdf"
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                str(D / "presentacion.html")], stderr=subprocess.DEVNULL, check=True)
from pypdf import PdfReader  # noqa: E402

print(f"Presentación: {out.name} | {len(PdfReader(out).pages)} diapositivas")

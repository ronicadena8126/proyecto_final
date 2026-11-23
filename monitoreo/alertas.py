"""
Monitoreo por lote (kestra/monitoreo.yml corre esto después de cada lote):

  python -m monitoreo.alertas --lote 7

1. Calcula métricas de SISTEMA (latencia p50/p95, 4xx, 5xx, volumen) del lote 7,
   de MODELO (distribución de probabilidades, tasa de cada decisión, respaldo),
   de DATOS (faltantes, fuera de rango, categorías no vistas) y la DERIVA (PSI, KS, chi²) del lote 7
   contra entrenamiento; y el DESEMPEÑO (AUC, ahorro vs. regla) del lote 6, cuyo target acaba de llegar.
2. Guarda todo en MONITOREO (formato largo: una fila por lote, métrica y feature).
3. Evalúa las REGLAS de alerta (plantilla de la sección 6.3), guarda las que se disparan en ALERTAS y
   las escribe en el log (Kestra las muestra en ERROR/WARN).
4. Decide si hay que reentrenar (sección 7.1) y lo devuelve a Kestra:
     - por desempeño: AUC o ahorro bajo el umbral en 2 lotes etiquetados SEGUIDOS del champion actual;
     - por calendario: 5 lotes desde el último reentrenamiento, aunque no haya alertas;
     - la deriva de datos sin caída de desempeño NO dispara reentrenamiento.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from monitoreo import deriva
from src import almacen, negocio
from src.contrato import FEATURES
from src.datos import ID

N_CONSECUTIVOS = 2          # lotes degradados seguidos para reentrenar por desempeño
CALENDARIO_LOTES = 5        # reentrenamiento por calendario cada 5 lotes sin reentrenar
Z_CONCEPTO = 2.0            # AUC del lote < media CV - 2 sigma_lote
LAT_P95_MS = 250.0
TASA_5XX = 0.01
TASA_4XX = 0.02
PCT_FALTANTES = 0.01
PSI_SCORE = 0.25

# Plantilla de la sección 6.3 (métrica, umbral, ventana, severidad, quién la recibe, acción).
REGLAS = [
    {"regla": "deriva_datos_feature", "metrica": "PSI por feature", "umbral": "> 0,25", "ventana": "1 lote",
     "severidad": "Alta", "destinatario": "Equipo de datos",
     "accion": "Revisar la fuente y el cambio de negocio; evaluar reentrenamiento si cae el desempeño"},
    {"regla": "deriva_datos_moderada", "metrica": "PSI por feature", "umbral": "0,10-0,25 en >= 3 features",
     "ventana": "1 lote", "severidad": "Media", "destinatario": "Equipo de datos",
     "accion": "Vigilar el siguiente lote; no se actúa todavía"},
    {"regla": "deriva_score", "metrica": "PSI de la probabilidad", "umbral": "> 0,25", "ventana": "1 lote",
     "severidad": "Media", "destinatario": "Equipo de ciencia de datos",
     "accion": "Revisar la tasa de contacto y la capacidad del call center"},
    {"regla": "capacidad_excedida", "metrica": "Tasa de 'contactar'", "umbral": "> 25% (capacidad 20%)",
     "ventana": "1 lote", "severidad": "Media", "destinatario": "Gerencia de cobranza",
     "accion": "Llamar solo al top 20% por probabilidad; recalibrar el corte con el lote más reciente"},
    {"regla": "concepto_auc", "metrica": "AUC del lote etiquetado", "umbral": "< media CV - 2 sigma_lote",
     "ventana": "1 lote (reentrena con 2 seguidos)", "severidad": "Alta", "destinatario": "Equipo de ciencia de datos",
     "accion": "Diagnosticar deriva conceptual; 2 lotes seguidos disparan el reentrenamiento"},
    {"regla": "negocio_bajo_regla", "metrica": "Ahorro del modelo vs. regla (igual volumen)", "umbral": "< ahorro de la regla",
     "ventana": "1 lote (reentrena con 2 seguidos)", "severidad": "Alta", "destinatario": "Gerencia de cobranza y ciencia de datos",
     "accion": "Evaluar reentrenamiento; si persiste después de reentrenar, pasar a respaldo (criterio de retirada)"},
    {"regla": "latencia_p95", "metrica": "Latencia p95 de /predecir", "umbral": f"> {LAT_P95_MS:.0f} ms",
     "ventana": "1 lote", "severidad": "Media", "destinatario": "Equipo de plataforma",
     "accion": "Revisar recursos del contenedor y el tamaño del lote"},
    {"regla": "errores_5xx", "metrica": "Tasa de errores 5xx", "umbral": f"> {TASA_5XX:.0%}", "ventana": "1 lote",
     "severidad": "Alta", "destinatario": "Equipo de plataforma",
     "accion": "Revisar logs de la API; si el modelo falla, activar el interruptor de respaldo"},
    {"regla": "errores_4xx", "metrica": "Tasa de rechazos 4xx", "umbral": f"> {TASA_4XX:.0%}", "ventana": "1 lote",
     "severidad": "Media", "destinatario": "Equipo de datos",
     "accion": "Revisar el sistema de origen: llegan registros que no cumplen el contrato"},
    {"regla": "calidad_faltantes", "metrica": "% faltantes por feature", "umbral": f"> {PCT_FALTANTES:.0%}",
     "ventana": "1 lote", "severidad": "Media", "destinatario": "Equipo de datos",
     "accion": "Revisar la carga del core; corregir antes del siguiente lote"},
    {"regla": "respaldo_activo", "metrica": "% de predicciones con respaldo", "umbral": "> 0%", "ventana": "1 lote",
     "severidad": "Alta", "destinatario": "Equipo de plataforma y gerencia de cobranza",
     "accion": "Confirmar si el respaldo fue manual; si no, revisar la carga del modelo"},
]
REGLAS_POR_ID = {r["regla"]: r for r in REGLAS}


def _out(**kv):
    print("::" + json.dumps({"outputs": kv}, default=str) + "::")


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- referencias
def referencia_entrenamiento() -> pd.DataFrame:
    from src.datos import cargar_obt, particion

    df, _ = cargar_obt()
    train, _ = particion(df)
    return train


def info_modelo(version: int) -> dict:
    """Media y desviación de la CV y saldo medio de la versión que produjo las predicciones."""
    from src.registro import cliente
    from src.tracking import MODELO_REGISTRO

    c = cliente()
    mv = c.get_model_version(MODELO_REGISTRO, str(int(version)))
    run = c.get_run(mv.run_id)
    m, p = run.data.metrics, run.data.params
    return {"version": int(version), "run_id": mv.run_id, "cv_mean": m.get("cv_roc_auc_mean"),
            "cv_std": m.get("cv_roc_auc_std"), "n_fold": float(p.get("n_entrenamiento", 24000)) / 5,
            "saldo_medio": float(p.get("saldo_medio_NT", negocio.SALDO_MEDIO))}


def sigma_lote(cv_std: float, n_fold: float, y: np.ndarray, s: np.ndarray, semilla: int = 0) -> float:
    """Variación esperada del AUC en un lote: la mayor entre la desviación de la CV llevada al tamaño
    del lote (cv_std * sqrt(n_fold / n_lote)) y el error estándar bootstrap del propio lote."""
    escalada = cv_std * np.sqrt(n_fold / len(y))
    rng = np.random.default_rng(semilla)
    boot = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boot.append(roc_auc_score(y[i], s[i]))
    return float(max(escalada, np.std(boot)))


# ----------------------------------------------------------------------------- métricas
def metricas_sistema(lote: int) -> dict:
    if not almacen.existe("SOLICITUDES"):
        return {}
    s = almacen.leer(f"""SELECT status, latencia_ms FROM {almacen.ESQUEMA}.SOLICITUDES
                         WHERE lote_id = {lote} AND ruta = '/predecir'""")
    if s.empty:
        return {}
    return {"volumen_solicitudes": float(len(s)), "latencia_p50_ms": float(s.latencia_ms.quantile(.5)),
            "latencia_p95_ms": float(s.latencia_ms.quantile(.95)),
            "tasa_4xx": float(((s.status >= 400) & (s.status < 500)).mean()),
            "tasa_5xx": float((s.status >= 500).mean())}


def metricas_modelo(lote: int, train: pd.DataFrame) -> tuple[dict, int | None]:
    if not almacen.existe("PREDICCIONES"):
        return {}, None
    p = almacen.leer(f"""SELECT probabilidad, decision, respaldo, version_modelo FROM {almacen.ESQUEMA}.PREDICCIONES
                         WHERE lote_id = {lote}""")
    if p.empty:
        return {}, None
    versiones = p.version_modelo.dropna()
    version = int(versiones.mode().iloc[0]) if len(versiones) else None
    m = {"n_predicciones": float(len(p)), "prob_media": float(p.probabilidad.mean()),
         "prob_p10": float(p.probabilidad.quantile(.1)), "prob_p50": float(p.probabilidad.quantile(.5)),
         "prob_p90": float(p.probabilidad.quantile(.9)),
         "tasa_contactar": float((p.decision == "contactar").mean()),
         "tasa_no_contactar": float((p.decision == "no_contactar").mean()),
         "tasa_respaldo": float(p.respaldo.astype(float).mean())}
    if version is not None:
        try:   # distribución de la probabilidad vs. la del mismo modelo sobre entrenamiento
            import mlflow

            from src.tracking import MODELO_REGISTRO, configurar

            configurar()
            mod = mlflow.sklearn.load_model(f"models:/{MODELO_REGISTRO}/{version}")
            ref = pd.Series(mod.predict_proba(train[FEATURES])[:, 1])
            m["psi_probabilidad"] = deriva.psi(ref, p.probabilidad[p.respaldo.astype(int) == 0])
        except Exception as e:
            print("  (no se pudo calcular el PSI de la probabilidad:", e, ")")
    return m, version


def desempeno(lote_etiquetado: int, lotes: pd.DataFrame | None = None) -> dict | None:
    """AUC y ahorro del lote cuyo target llegó. Ahorro a igual capacidad que en producción."""
    if not almacen.existe("DESEMPENO"):
        return None
    d = almacen.leer(f"SELECT * FROM {almacen.ESQUEMA}.DESEMPENO WHERE lote_id = {lote_etiquetado}")
    d = d[d.respaldo.astype(int) == 0] if (d.respaldo.astype(int) == 0).any() else d
    if d.empty or d.target.nunique() < 2:
        return None
    x = almacen.leer(f"""SELECT {ID} AS customer_id, pay_status_m1 FROM {almacen.ESQUEMA}.LOTES
                         WHERE lote_id = {lote_etiquetado}""")
    d = d.merge(x, on="customer_id", how="left")
    version = int(d.version_modelo.dropna().mode().iloc[0]) if d.version_modelo.notna().any() else None
    info = info_modelo(version) if version is not None else {"cv_mean": None, "cv_std": None, "n_fold": 4800,
                                                             "saldo_medio": negocio.SALDO_MEDIO}
    y, s = d.target.to_numpy().astype(int), d.probabilidad.to_numpy()
    saldo = info["saldo_medio"]
    # Política operativa: contactar a quien supera el corte, hasta la capacidad (20% del lote).
    contactados = (d.decision == "contactar").to_numpy() & negocio.contactar_top(s, 0.0)
    regla = (d.pay_status_m1.fillna(-2) >= 1).to_numpy()
    k_regla = int(regla.sum())
    out = {"version_modelo": version, "n_etiquetados": float(len(d)), "tasa_impago_real": float(y.mean()),
           "auc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
           "recall_top20": negocio.recall_precision_top(y, s)[0],
           "contactados": float(contactados.sum()), "moras_detectadas": float((contactados & (y == 1)).sum()),
           "ahorro_modelo": negocio.ahorro(y, contactados, saldo),
           "ahorro_regla": negocio.ahorro(y, regla, saldo),
           # comparación justa (PSet 3): el modelo llama a la MISMA cantidad de clientes que la regla
           "ahorro_modelo_igual_volumen": negocio.ahorro(y, negocio.contactar_k(s, k_regla), saldo),
           "ahorro_nada": 0.0, "contactados_regla": float(k_regla)}
    if info.get("cv_mean") is not None:
        sig = sigma_lote(info["cv_std"], info["n_fold"], y, s)
        out.update({"cv_mean": info["cv_mean"], "cv_std": info["cv_std"], "sigma_lote": sig,
                    "umbral_auc": info["cv_mean"] - Z_CONCEPTO * sig})
    return out


# ----------------------------------------------------------------------------- reglas de alerta
def evaluar(lote: int, sis: dict, mod: dict, der: pd.DataFrame, cal: pd.DataFrame,
            des: dict | None, lote_et: int) -> list[dict]:
    alertas = []

    def alerta(regla, valor, detalle, lote_ref=lote, feature=""):
        r = REGLAS_POR_ID[regla]
        alertas.append({"lote_id": lote, "lote_referido": lote_ref, "regla": regla, "metrica": r["metrica"],
                        "feature": feature, "valor": float(valor), "umbral": r["umbral"], "ventana": r["ventana"],
                        "severidad": r["severidad"], "destinatario": r["destinatario"], "accion": r["accion"],
                        "mensaje": detalle, "ts": _ahora()})

    if len(der):
        altas = der[der.nivel == "alta"]
        if len(altas):   # una alerta por lote; el PSI de cada feature queda en MONITOREO
            f = altas.iloc[0]
            alerta("deriva_datos_feature", f.psi, f"{len(altas)} features con PSI > 0,25: "
                   + ", ".join(f"{r.feature}={r.psi:.2f}" for _, r in altas.iterrows())
                   + f" (máx. {f.feature}; {f.prueba} p={f.p_valor:.1e})", feature=f.feature)
        mods = der[der.nivel == "moderada"]
        if len(altas) == 0 and len(mods) >= 3:
            alerta("deriva_datos_moderada", len(mods), f"{len(mods)} features con PSI 0,10-0,25: "
                   + ", ".join(mods.feature.head(5)))
    if mod.get("psi_probabilidad", 0) > PSI_SCORE:
        alerta("deriva_score", mod["psi_probabilidad"], f"PSI de la probabilidad = {mod['psi_probabilidad']:.3f}")
    if mod.get("tasa_contactar", 0) > negocio.CAPACIDAD * 1.25:
        alerta("capacidad_excedida", mod["tasa_contactar"], f"{mod['tasa_contactar']:.0%} de la cartera supera el corte "
               f"y la capacidad es {negocio.CAPACIDAD:.0%}: cobranza llama solo al top {negocio.CAPACIDAD:.0%} por riesgo")
    if mod.get("tasa_respaldo", 0) > 0:
        alerta("respaldo_activo", mod["tasa_respaldo"], f"{mod['tasa_respaldo']:.0%} de las predicciones usaron la regla")
    if sis:
        if sis["latencia_p95_ms"] > LAT_P95_MS:
            alerta("latencia_p95", sis["latencia_p95_ms"], f"p95 = {sis['latencia_p95_ms']:.0f} ms")
        if sis["tasa_5xx"] > TASA_5XX:
            alerta("errores_5xx", sis["tasa_5xx"], f"5xx = {sis['tasa_5xx']:.1%}")
        if sis["tasa_4xx"] > TASA_4XX:
            alerta("errores_4xx", sis["tasa_4xx"], f"4xx = {sis['tasa_4xx']:.1%}")
    if len(cal):
        for _, f in cal[cal.pct_faltantes > PCT_FALTANTES].iterrows():
            alerta("calidad_faltantes", f.pct_faltantes, f"{f.feature}: {f.pct_faltantes:.1%} faltantes", feature=f.feature)
    if des and "umbral_auc" in des and des["auc"] < des["umbral_auc"]:
        alerta("concepto_auc", des["auc"], f"AUC lote {lote_et} = {des['auc']:.3f} < {des['umbral_auc']:.3f} "
               f"(CV {des['cv_mean']:.3f} - {Z_CONCEPTO:.0f}x{des['sigma_lote']:.3f})", lote_ref=lote_et)
    if des and des["ahorro_modelo_igual_volumen"] < des["ahorro_regla"]:
        alerta("negocio_bajo_regla", des["ahorro_modelo_igual_volumen"],
               f"Ahorro lote {lote_et}: modelo NT$ {des['ahorro_modelo_igual_volumen']:,.0f} < regla "
               f"NT$ {des['ahorro_regla']:,.0f} (mismo número de llamadas)", lote_ref=lote_et)
    return alertas


# ----------------------------------------------------------------------------- disparador
def degradado(des: dict | None) -> bool:
    if not des:
        return False
    auc_mal = "umbral_auc" in des and des["auc"] < des["umbral_auc"]
    negocio_mal = des["ahorro_modelo_igual_volumen"] < des["ahorro_regla"]
    return bool(auc_mal or negocio_mal)


def decidir_reentrenamiento(lote: int, hay_lote_nuevo: bool) -> tuple[bool, str, str]:
    """Devuelve (reentrenar, motivo, explicación)."""
    if not hay_lote_nuevo:
        return False, "", "cierre: no llega un lote nuevo, el disparador no se evalúa"
    hechos = pd.DataFrame()
    if almacen.existe("REENTRENAMIENTOS"):
        hechos = almacen.leer(f"SELECT lote_disparo, decision FROM {almacen.ESQUEMA}.REENTRENAMIENTOS")
        if (hechos.lote_disparo == lote).any():
            return False, "", f"ya hubo un reentrenamiento en el lote {lote}"
    from src.registro import version_por_alias

    champion = int(version_por_alias("champion").version)
    mon = almacen.leer(f"""SELECT lote_id, metrica, valor FROM {almacen.ESQUEMA}.MONITOREO
                           WHERE tipo = 'desempeno' AND metrica IN ('auc','umbral_auc','version_modelo',
                           'ahorro_modelo_igual_volumen','ahorro_regla')""")
    if len(mon):
        piv = mon.pivot_table(index="lote_id", columns="metrica", values="valor").sort_index()
        piv = piv[piv.get("version_modelo") == champion] if "version_modelo" in piv else piv.iloc[0:0]
        ult = piv.tail(N_CONSECUTIVOS)
        if len(ult) == N_CONSECUTIVOS and list(ult.index) == list(range(ult.index.max() - N_CONSECUTIVOS + 1,
                                                                          ult.index.max() + 1)):
            malos = [degradado({"auc": r.auc, "umbral_auc": r.umbral_auc, "ahorro_regla": r.ahorro_regla,
                                "ahorro_modelo_igual_volumen": r.ahorro_modelo_igual_volumen})
                     for _, r in ult.iterrows()]
            if all(malos):
                return True, "desempeno", (f"lotes {list(ult.index)} del champion v{champion} bajo el umbral "
                                           f"(AUC {', '.join(f'{a:.3f}' for a in ult.auc)})")
    ultimo = int(hechos.lote_disparo.max()) if len(hechos) else 0
    if lote - ultimo >= CALENDARIO_LOTES:
        return True, "calendario", f"{lote - ultimo} lotes desde el último reentrenamiento (cada {CALENDARIO_LOTES})"
    return False, "", "sin disparador (la deriva de datos sola no reentrena)"


# ----------------------------------------------------------------------------- principal
def monitorear(lote: int) -> dict:
    lote = int(lote)
    lote_et = lote - 1
    train = referencia_entrenamiento()
    feats = FEATURES
    act = almacen.leer(f"SELECT * FROM {almacen.ESQUEMA}.LOTES WHERE lote_id = {lote}")
    hay_lote = not act.empty

    sis = metricas_sistema(lote) if hay_lote else {}
    mod, version = metricas_modelo(lote, train) if hay_lote else ({}, None)
    der = deriva.deriva_lote(train, act, feats) if hay_lote else pd.DataFrame()
    cal = deriva.calidad_lote(train, act, feats) if hay_lote else pd.DataFrame()
    des = desempeno(lote_et) if lote_et >= 1 else None

    filas = []

    def add(tipo, metrica, valor, feature="", umbral=None, estado="ok", lote_id=lote):
        filas.append({"lote_id": lote_id, "evaluado_en_lote": lote, "tipo": tipo, "metrica": metrica,
                      "feature": feature, "valor": None if valor is None else float(valor),
                      "umbral": umbral, "estado": estado, "version_modelo": version, "ts": _ahora()})

    for k, v in sis.items():
        add("sistema", k, v)
    for k, v in mod.items():
        add("modelo", k, v)
    for _, f in der.iterrows():
        add("deriva", "psi", f.psi, f.feature, deriva.PSI_ALTA, f.nivel)
        add("deriva", f"{f.prueba}_p_valor", f.p_valor, f.feature, 0.01, "confirma" if f.prueba_confirma else "ok")
    if len(der):
        add("deriva", "psi_max", der.psi.max(), der.feature.iloc[0], deriva.PSI_ALTA, deriva.nivel_psi(der.psi.max()))
        add("deriva", "n_features_psi_alta", (der.nivel == "alta").sum())
    for _, f in cal.iterrows():
        add("datos", "pct_faltantes", f.pct_faltantes, f.feature)
        add("datos", "pct_fuera_rango", f.pct_fuera_rango, f.feature)
        add("datos", "pct_categorias_no_vistas", f.pct_categorias_no_vistas, f.feature)
    if des:
        for k, v in des.items():
            if v is not None:
                add("desempeno", k, v, lote_id=lote_et)
        # diagnóstico de la tabla 6.2 para el lote etiquetado (entradas de ese lote + su desempeño)
        psi_prev = almacen.leer(f"""SELECT valor FROM {almacen.ESQUEMA}.MONITOREO WHERE lote_id = {lote_et}
                                    AND metrica = 'n_features_psi_alta'""") if almacen.existe("MONITOREO") else pd.DataFrame()
        cambian = bool(len(psi_prev) and psi_prev.valor.iloc[0] > 0)
        cae = degradado(des)
        diag = {(True, False): "deriva_datos_sin_impacto", (True, True): "deriva_datos_con_impacto",
                (False, True): "deriva_conceptual", (False, False): "estable"}[(cambian, cae)]
        add("diagnostico", diag, 1.0, estado=diag, lote_id=lote_et)

    alertas = evaluar(lote, sis, mod, der, cal, des, lote_et)
    reentrenar, motivo, porque = False, "", ""
    # MONITOREO se escribe antes de decidir: el disparador mira la historia completa.
    mon = pd.DataFrame(filas)
    if almacen.existe("MONITOREO"):
        almacen.ejecutar(f"DELETE FROM {almacen.ESQUEMA}.MONITOREO WHERE evaluado_en_lote = {lote}")
    almacen.escribir("MONITOREO", mon)
    reentrenar, motivo, porque = decidir_reentrenamiento(lote, hay_lote)
    if almacen.existe("ALERTAS"):
        almacen.ejecutar(f"DELETE FROM {almacen.ESQUEMA}.ALERTAS WHERE lote_id = {lote}")
    almacen.escribir("ALERTAS", pd.DataFrame(alertas))

    # --- log legible (Kestra) ---
    print(f"=== Monitoreo lote {lote} (desempeño del lote {lote_et}) ===")
    if sis:
        print(f"Sistema: {sis['volumen_solicitudes']:.0f} solicitudes | p50 {sis['latencia_p50_ms']:.1f} ms | "
              f"p95 {sis['latencia_p95_ms']:.1f} ms | 4xx {sis['tasa_4xx']:.1%} | 5xx {sis['tasa_5xx']:.1%}")
    if mod:
        print(f"Modelo v{version}: prob. media {mod['prob_media']:.3f} | contactar {mod['tasa_contactar']:.1%} | "
              f"respaldo {mod['tasa_respaldo']:.0%} | PSI prob. {mod.get('psi_probabilidad', float('nan')):.3f}")
    if len(der):
        print("Deriva (top 5 PSI): " + ", ".join(f"{r.feature}={r.psi:.3f}" for _, r in der.head(5).iterrows()))
    if des:
        print(f"Desempeño lote {lote_et}: AUC {des['auc']:.3f} (umbral {des.get('umbral_auc', float('nan')):.3f}) | "
              f"ahorro modelo NT$ {des['ahorro_modelo']:,.0f} vs regla NT$ {des['ahorro_regla']:,.0f}")
    for a in alertas:
        print(f"ALERTA [{a['severidad']}] {a['regla']}: {a['mensaje']} -> {a['destinatario']}: {a['accion']}")
    if not alertas:
        print("Sin alertas")
    print(f"Reentrenar: {reentrenar} ({motivo or '-'}): {porque}")
    resumen = "; ".join(f"[{a['severidad']}] {a['mensaje']}" for a in alertas)[:2000]
    _out(lote=lote, n_alertas=len(alertas), alertas=resumen, reentrenar=reentrenar, motivo=motivo,
         explicacion=porque, n_alertas_altas=sum(a["severidad"] == "Alta" for a in alertas))
    return {"alertas": alertas, "reentrenar": reentrenar, "motivo": motivo}


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", type=int, required=True)
    monitorear(ap.parse_args().lote)


if __name__ == "__main__":
    _main()

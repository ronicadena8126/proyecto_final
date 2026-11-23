"""
Pasos de la simulación de producción que orquesta kestra/produccion_lotes.yml (uno por tarea):

  python -m src.produccion preparar            # notebook 06: sube LOTES y TARGETS al almacén
  python -m src.produccion puntuar --lote 3    # lee el lote, llama a la API cliente por cliente
  python -m src.produccion volcar              # registro de la API -> PREDICCIONES y SOLICITUDES
  python -m src.produccion unir --lote 3       # targets que ya llegaron -> DESEMPENO (lote 2)

Patrón batch: el día 1 de cada mes se puntúa la cartera del mes y cobranza recibe la lista. La API
se usa igual que la usaría cualquier otro consumidor (validación, registro, versión del modelo).
Todos los pasos son idempotentes: reprocesar un lote borra y vuelve a escribir solo ese lote.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src import almacen
from src.datos import ID, TARGET

API_URL = os.environ.get("API_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
ENTEROS = {"education_code", "is_education_imputed", "is_marriage_imputed", "age", "is_inactive_6m",
           "months_with_credit_balance", "max_delay_6m", "n_delinquent_months_6m",
           "n_delinquent_months_3m", "n_overlimit_months_6m", "n_credit_balance_months_6m",
           "n_inconsistent_months_6m", "is_delinquent_last_month", ID} | {f"pay_status_m{m}" for m in range(1, 7)}
NO_ENVIAR = {"lote_id", "periodo", "deriva_inducida", "registro_corrupto", "customer_key",
             "attributes_as_of", "observation_month_key", "label_month_key", "n_months_history",
             "education", "age_band", "avg_payment_ratio_6m", "datos_hash_obt"}


def _out(**kv):
    """Salidas para Kestra (outputs.<tarea>.vars.<clave>) y log legible."""
    print("::" + json.dumps({"outputs": kv}, default=str) + "::")


# ----------------------------------------------------------------------------- preparar
def preparar(n_lotes: int | None = None, rehacer: bool = False) -> dict:
    """Construye y sube LOTES y TARGETS. Si ya hay una simulación en curso o terminada, NO la borra
    (salvo rehacer=True): así volver a ejecutar el notebook 06 no destruye la evidencia."""
    from src import simulacion
    from src.datos import cargar_obt, particion

    almacen.preparar()
    if not rehacer and almacen.existe("LOTES") and almacen.existe("PREDICCIONES"):
        n = almacen.leer(f"SELECT COUNT(*) AS n, COUNT(DISTINCT lote_id) AS l FROM {almacen.ESQUEMA}.LOTES")
        print(f"Ya existe una simulación ({int(n.l.iloc[0])} lotes, {int(n.n.iloc[0])} filas): no se toca. "
              "Para empezar de cero: python -m src.produccion preparar --rehacer")
        return {"lotes": int(n.l.iloc[0]), "rehecho": False}
    df, meta = cargar_obt()
    _, test = particion(df)
    lotes, targets = simulacion.construir_lotes(test, n_lotes or simulacion.N_LOTES)
    lotes["datos_hash_obt"] = meta["hash_dataset"]
    n1 = almacen.escribir("LOTES", lotes, reemplazar=True)
    n2 = almacen.escribir("TARGETS", targets, reemplazar=True)
    for t in ("PREDICCIONES", "SOLICITUDES", "DESEMPENO", "MONITOREO", "ALERTAS", "REENTRENAMIENTOS",
              "CONTROL_VOLCADO"):
        if almacen.existe(t):
            almacen.ejecutar(f"DROP TABLE {almacen.ESQUEMA}.{t}")
    res = {"lotes": int(lotes.lote_id.nunique()), "filas_lotes": n1, "filas_targets": n2,
           "hash_obt": meta["hash_dataset"]}
    print(res)
    return res


# ----------------------------------------------------------------------------- puntuar
def _fila_json(fila: pd.Series) -> dict:
    out = {}
    for c, v in fila.items():
        if c in NO_ENVIAR:
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[c] = None
        elif c in ENTEROS:
            out[c] = int(v)
        else:
            out[c] = float(v)
    return out


def puntuar(lote: int) -> dict:
    import httpx

    df = almacen.leer(f"SELECT * FROM {almacen.ESQUEMA}.LOTES WHERE lote_id = {int(lote)} ORDER BY {ID}")
    if df.empty:
        print(f"Lote {lote}: sin clientes (cierre de la simulación); no hay nada que puntuar")
        _out(lote=lote, enviados=0, ok=0, rechazados=0, errores=0)
        return {"enviados": 0}
    # idempotencia: si el lote se reprocesa, se reemplazan sus filas en PREDICCIONES / SOLICITUDES
    for t in ("PREDICCIONES", "SOLICITUDES"):
        if almacen.existe(t):
            almacen.ejecutar(f"DELETE FROM {almacen.ESQUEMA}.{t} WHERE lote_id = {int(lote)}")
    ok = rech = err = 0
    t0 = time.perf_counter()
    with httpx.Client(base_url=API_URL, timeout=30) as cli:
        cli.get("/salud").raise_for_status()
        for _, fila in df.iterrows():
            r = cli.post("/predecir", json=_fila_json(fila), headers={"X-Lote": str(lote)})
            if r.status_code == 200:
                ok += 1
            elif 400 <= r.status_code < 500:
                rech += 1
                print(f"  rechazado customer_id={int(fila[ID])}: {r.json()['detail']}")
            else:
                err += 1
    seg = time.perf_counter() - t0
    print(f"Lote {lote}: {len(df)} enviados, {ok} puntuados, {rech} rechazados (4xx), {err} errores (5xx) en {seg:.1f}s")
    _out(lote=lote, enviados=len(df), ok=ok, rechazados=rech, errores=err)
    return {"enviados": len(df), "ok": ok, "rechazados": rech, "errores": err}


# ----------------------------------------------------------------------------- volcar
def _marca(tabla: str) -> int:
    if not almacen.existe("CONTROL_VOLCADO"):
        return 0
    d = almacen.leer(f"SELECT max_id FROM {almacen.ESQUEMA}.CONTROL_VOLCADO WHERE tabla = '{tabla}'")
    return int(d.max_id.iloc[0]) if len(d) else 0


def volcar() -> dict:
    """Carga periódica: lo que la API registró desde la última marca de agua."""
    import httpx

    res = {}
    with httpx.Client(base_url=API_URL, timeout=120, headers={"X-Token": ADMIN_TOKEN}) as cli:
        for tabla in ("predicciones", "solicitudes"):
            desde = _marca(tabla)
            r = cli.get("/admin/pendientes", params={"tabla": tabla, "desde_id": desde})
            r.raise_for_status()
            j = r.json()
            if j["ultimo_id"] < desde:
                # el volumen de la API se recreó (ids desde 1 otra vez): se reinicia la marca de agua
                print(f"  {tabla}: el registro de la API se reinició; se reinicia la marca de agua")
                j = cli.get("/admin/pendientes", params={"tabla": tabla, "desde_id": 0}).json()
            df = pd.DataFrame(j["filas"])
            if len(df):
                df = df.rename(columns={"id": "api_id"})
                almacen.escribir(tabla.upper(), df)
                marca = pd.DataFrame([{"tabla": tabla, "max_id": int(df.api_id.max()),
                                       "actualizado": pd.Timestamp.utcnow().isoformat()}])
                almacen.escribir("CONTROL_VOLCADO", marca, borrar_donde=f"tabla = '{tabla}'")
            res[tabla] = len(df)
            print(f"  {tabla.upper()}: {len(df)} filas nuevas")
    _out(**res)
    return res


# ----------------------------------------------------------------------------- unir
def unir(lote: int) -> dict:
    """Une predicciones con los targets que ya llegaron (respeta el horizonte de 1 lote)."""
    disp = almacen.leer(f"""SELECT DISTINCT lote_id FROM {almacen.ESQUEMA}.TARGETS
                            WHERE disponible_en_lote <= {int(lote)}""").lote_id.tolist()
    hechos = []
    if almacen.existe("DESEMPENO"):
        hechos = almacen.leer(f"SELECT DISTINCT lote_id FROM {almacen.ESQUEMA}.DESEMPENO").lote_id.tolist()
    nuevos = sorted(set(disp) - set(hechos))
    if lote - 1 in disp and lote - 1 not in nuevos:
        nuevos.append(lote - 1)              # reproceso del lote anterior: se vuelve a unir
    total = 0
    for k in nuevos:
        p = almacen.leer(f"""SELECT prediccion_id, ts, lote_id, customer_id, probabilidad, decision, umbral,
                                    fuente, respaldo, version_modelo, run_id
                             FROM {almacen.ESQUEMA}.PREDICCIONES WHERE lote_id = {int(k)}""")
        t = almacen.leer(f"""SELECT {ID} AS customer_id, {TARGET} AS target, target_alterado, disponible_en_lote
                             FROM {almacen.ESQUEMA}.TARGETS WHERE lote_id = {int(k)}""")
        if p.empty:
            continue
        p = p.sort_values("ts").drop_duplicates("customer_id", keep="last")
        d = p.merge(t, on="customer_id", how="inner")
        d["unido_en_lote"] = int(lote)
        total += almacen.escribir("DESEMPENO", d, borrar_donde=f"lote_id = {int(k)}")
        print(f"  lote {k}: {len(d)} predicciones con target (llegó en el lote {lote})")
    if not nuevos:
        print(f"Lote {lote}: todavía no llega ningún target nuevo")
    _out(lote=lote, lotes_unidos=nuevos, filas=total)
    return {"lotes_unidos": nuevos, "filas": total}


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=["preparar", "puntuar", "volcar", "unir"])
    ap.add_argument("--lote", type=int)
    ap.add_argument("--n-lotes", type=int)
    ap.add_argument("--rehacer", action="store_true", help="borra la simulación anterior y empieza de cero")
    a = ap.parse_args()
    if a.paso == "preparar":
        preparar(a.n_lotes, a.rehacer)
    elif a.paso == "puntuar":
        puntuar(a.lote)
    elif a.paso == "volcar":
        volcar()
    else:
        unir(a.lote)


if __name__ == "__main__":
    _main()

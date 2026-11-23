"""
Model Registry de MLflow como fuente de verdad del modelo en producción (sección 4).

  models:/anticipar-mora-modelo@champion    -> lo que sirve la API
  models:/anticipar-mora-modelo@challenger  -> candidato a reemplazarlo
  models:/anticipar-mora-modelo@anterior    -> el champion previo (para rollback inmediato)

Uso por línea de comandos (rollback sin reconstruir la imagen de la API):
  python -m src.registro estado
  python -m src.registro rollback                # champion <- @anterior
  python -m src.registro asignar --version 1     # champion <- versión 1 (y @anterior <- el actual)
"""
from __future__ import annotations

import argparse
import os

from src.tracking import MODELO_REGISTRO


def cliente():
    import mlflow
    from mlflow import MlflowClient

    from src.tracking import configurar

    configurar()
    return MlflowClient(mlflow.get_tracking_uri())


def version_por_alias(alias: str = "champion", c=None):
    c = c or cliente()
    return c.get_model_version_by_alias(MODELO_REGISTRO, alias)


def info_version(alias: str = "champion", c=None) -> dict:
    """Versión, run_id y parámetros de negocio guardados en la corrida que produjo el modelo."""
    c = c or cliente()
    mv = version_por_alias(alias, c)
    run = c.get_run(mv.run_id)
    p = run.data.params
    return {
        "nombre": MODELO_REGISTRO, "alias": alias, "version": int(mv.version), "run_id": mv.run_id,
        "umbral_decision": float(p.get("umbral_decision", 0.05)),
        "corte_efectivo": float(p.get("corte_efectivo_train_oof", 0.34)),
        "saldo_medio": float(p.get("saldo_medio_NT", 47966.81)),
        "cv_roc_auc_mean": run.data.metrics.get("cv_roc_auc_mean"),
        "cv_roc_auc_std": run.data.metrics.get("cv_roc_auc_std"),
        "datos_hash": p.get("datos_hash"), "datos_tabla_origen": p.get("datos_tabla_origen"),
        "fase": run.data.tags.get("fase"), "modelo": run.data.tags.get("modelo"),
    }


def asignar_champion(version: int | str, c=None, motivo: str = "") -> dict:
    """Mueve @champion a `version` y deja el champion previo en @anterior."""
    c = c or cliente()
    try:
        previo = version_por_alias("champion", c).version
    except Exception:
        previo = None
    if previo is not None and str(previo) != str(version):
        c.set_registered_model_alias(MODELO_REGISTRO, "anterior", previo)
    c.set_registered_model_alias(MODELO_REGISTRO, "champion", str(version))
    c.set_model_version_tag(MODELO_REGISTRO, str(version), "promovido_motivo", motivo[:250] or "manual")
    return {"champion": str(version), "anterior": previo}


def rollback(c=None) -> dict:
    c = c or cliente()
    anterior = version_por_alias("anterior", c).version
    return asignar_champion(anterior, c, motivo="rollback")


def estado(c=None) -> dict:
    c = c or cliente()
    rm = c.get_registered_model(MODELO_REGISTRO)
    return {"nombre": MODELO_REGISTRO, "aliases": dict(rm.aliases)}


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["estado", "rollback", "asignar"])
    ap.add_argument("--version")
    a = ap.parse_args()
    if a.accion == "estado":
        print(estado())
    elif a.accion == "rollback":
        print(rollback())
    else:
        print(asignar_champion(a.version, motivo="asignación manual"))
    if os.environ.get("API_URL"):
        _avisar_api()


def _avisar_api():
    """Pide a la API recargar el champion ya (si no, lo detecta sola en <= RECARGA_SEGUNDOS)."""
    import httpx

    try:
        r = httpx.post(f"{os.environ['API_URL']}/admin/recargar",
                       headers={"X-Token": os.environ.get("ADMIN_TOKEN", "")}, timeout=60)
        print("API:", r.status_code, r.text[:300])
    except Exception as e:  # la API también recarga sola
        print("API no disponible para recargar ahora:", e)


if __name__ == "__main__":
    _main()

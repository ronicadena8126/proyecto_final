"""Utilidades de MLflow: configuración, evaluación con CV y registro homogéneo de corridas."""
from __future__ import annotations

import os
import subprocess
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src.datos import RAIZ, SEMILLA, esquema_cv

EXPERIMENTO = "anticipar-mora"
MODELO_REGISTRO = "anticipar-mora-modelo"
METRICA = "roc_auc"                      # métrica técnica (misma para todos los modelos)
SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision"}


def configurar() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENTO)
    return uri


def commit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "sin-git"


def evaluar_cv(pipe, X, y, cv=None) -> dict:
    """CV estratificada con métricas de entrenamiento y validación."""
    cv = cv or esquema_cv()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=1)
    total = time.perf_counter() - t0
    out = {}
    for m in SCORING:
        out[f"train_{m}"] = float(np.mean(r[f"train_{m}"]))
        out[f"cv_{m}_mean"] = float(np.mean(r[f"test_{m}"]))
        out[f"cv_{m}_std"] = float(np.std(r[f"test_{m}"]))
    out["brecha_roc_auc"] = out["train_roc_auc"] - out["cv_roc_auc_mean"]
    out["fit_seconds_mean"] = float(np.mean(r["fit_time"]))
    out["cv_seconds_total"] = float(total)
    return out


def log_datos(meta: dict) -> None:
    mlflow.log_params({
        "datos_tabla_origen": meta["tabla_origen"],
        "datos_fecha_corte": meta["fecha_corte_snapshot"],
        "datos_hash": meta["hash_dataset"],
        "datos_n_filas": meta["n_filas"],
    })


def log_modelo(pipe, X_ejemplo: pd.DataFrame, nombre: str = "modelo", registrar: str | None = None):
    ejemplo = X_ejemplo.head(5)
    firma = infer_signature(ejemplo, pipe.predict_proba(ejemplo)[:, 1])
    return mlflow.sklearn.log_model(
        pipe, artifact_path=nombre, signature=firma, input_example=ejemplo,
        code_paths=[str(RAIZ / "src")], registered_model_name=registrar,
        pyfunc_predict_fn="predict_proba",
    )


def correr(nombre: str, fase: str, familia: str, pipe, X, y, meta: dict,
           params: dict | None = None, log_model: bool = True) -> dict:
    """Una corrida estándar: CV + ajuste final en todo el entrenamiento + registro completo."""
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags({"fase": fase, "familia": familia, "git_commit": commit_git(),
                         "modelo": nombre})
        log_datos(meta)
        p = {"semilla": SEMILLA, "cv": "StratifiedKFold(5, shuffle, 42)", "metrica": METRICA}
        modelo = pipe.steps[-1][1]
        p.update({f"hp_{k}": v for k, v in modelo.get_params().items()
                  if isinstance(v, (int, float, str, bool, type(None)))})
        if params:
            p.update(params)
        mlflow.log_params(p)
        res = evaluar_cv(pipe, X, y)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ajustado = clone(pipe).fit(X, y)
        res["train_seconds"] = time.perf_counter() - t0
        mlflow.log_metrics(res)
        if log_model:
            log_modelo(ajustado, X)
        res.update({"run_id": run.info.run_id, "modelo": nombre, "familia": familia, "fase": fase})
    return res

"""
Exporta el champion para las pruebas de CI (en GitHub Actions no hay servidor de MLflow):

  python -m src.exportar_fixture

  tests/fixtures/modelo/        mlflow.sklearn.save_model del champion (con su firma)
  tests/fixtures/muestra.csv    600 clientes fijos del conjunto de prueba (estratificado, semilla 42)
  tests/fixtures/referencia.json versión, run_id, umbral y AUC del modelo en esa muestra

El AUC en la muestra fija se registra TAMBIÉN en la corrida del champion en MLflow
(métrica auc_muestra_fija), así la prueba compara contra lo que quedó registrado.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlflow
from mlflow.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import registro
from src.contrato import FEATURES
from src.datos import ID, SEMILLA, TARGET, cargar_obt, particion
from src.tracking import MODELO_REGISTRO, configurar

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "tests" / "fixtures"
N_MUESTRA = 600


def main():
    configurar()
    c = registro.cliente()
    info = registro.info_version("champion", c)
    uri = f"models:/{MODELO_REGISTRO}/{info['version']}"
    modelo = mlflow.sklearn.load_model(uri)
    firma = Model.load(uri).signature

    df, meta = cargar_obt()
    _, test = particion(df)
    feats = FEATURES
    muestra, _ = train_test_split(test, train_size=N_MUESTRA, stratify=test[TARGET], random_state=SEMILLA)
    muestra = muestra.sort_values(ID)[[ID] + feats + [TARGET]]
    auc = float(roc_auc_score(muestra[TARGET], modelo.predict_proba(muestra[feats])[:, 1]))
    c.log_metric(info["run_id"], "auc_muestra_fija", auc)

    DESTINO.mkdir(parents=True, exist_ok=True)
    if (DESTINO / "modelo").exists():
        shutil.rmtree(DESTINO / "modelo")
    mlflow.sklearn.save_model(modelo, DESTINO / "modelo", signature=firma,
                              code_paths=[str(RAIZ / "src")], serialization_format="cloudpickle")
    muestra.to_csv(DESTINO / "muestra.csv", index=False)
    ref = {"nombre": MODELO_REGISTRO, "alias": "champion", "version": info["version"], "run_id": info["run_id"],
           "auc_muestra_fija": auc, "metrica_mlflow": "auc_muestra_fija", "tolerancia_auc": 1e-3,
           "cv_roc_auc_mean": info["cv_roc_auc_mean"], "cv_roc_auc_std": info["cv_roc_auc_std"],
           "corte_efectivo": info["corte_efectivo"], "umbral_decision": info["umbral_decision"],
           "datos_hash": info["datos_hash"], "datos_tabla_origen": info["datos_tabla_origen"],
           "n_muestra": N_MUESTRA, "tasa_impago_muestra": float(muestra[TARGET].mean())}
    (DESTINO / "referencia.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print(json.dumps(ref, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Configuración común: la API usa la copia del champion de tests/fixtures (no hay MLflow en CI)."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
FIXTURES = RAIZ / "tests" / "fixtures"
sys.path.insert(0, str(RAIZ))

os.environ.setdefault("MODELO_URI", str(FIXTURES / "modelo"))
os.environ.setdefault("ADMIN_TOKEN", "token-de-prueba")
os.environ.setdefault("RECARGA_SEGUNDOS", "3600")
os.environ["MODO_RESPALDO"] = "0"


@pytest.fixture(scope="session")
def referencia() -> dict:
    return json.loads((FIXTURES / "referencia.json").read_text())


@pytest.fixture(scope="session")
def muestra() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "muestra.csv")


@pytest.fixture(scope="session")
def modelo():
    import mlflow

    return mlflow.sklearn.load_model(str(FIXTURES / "modelo"))


@pytest.fixture()
def api(tmp_path, monkeypatch):
    """Cliente HTTP de la API con un registro de predicciones temporal y estado limpio."""
    from fastapi.testclient import TestClient

    import api.main as m

    monkeypatch.setattr(m, "REGISTRO_DB", tmp_path / "registro.db")
    monkeypatch.setattr(m, "MODELO_URI", str(FIXTURES / "modelo"))
    monkeypatch.setattr(m, "ESTADO", m.EstadoModelo())
    with TestClient(m.app) as c:
        yield c


def cliente_valido(fila: pd.Series | None = None) -> dict:
    from api.esquemas import Cliente

    if fila is None:
        return dict(Cliente.model_config["json_schema_extra"]["example"])
    from src.produccion import _fila_json

    d = _fila_json(fila.drop(labels=["default_next_month"], errors="ignore"))
    return d

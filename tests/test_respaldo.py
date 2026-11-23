"""Respaldo (fallback) e interruptor: si el modelo no carga, o si se activa el interruptor, la API
responde con la regla de negocio del PSet 3 y lo indica en la respuesta."""
import pytest
from fastapi.testclient import TestClient

from tests.conftest import cliente_valido

TOKEN = {"X-Token": "token-de-prueba"}


@pytest.fixture()
def api_sin_modelo(tmp_path, monkeypatch):
    import api.main as m

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns-vacio")
    monkeypatch.setattr(m, "REGISTRO_DB", tmp_path / "registro.db")
    monkeypatch.setattr(m, "MODELO_URI", "models:/modelo-que-no-existe@champion")
    monkeypatch.setattr(m, "ESTADO", m.EstadoModelo())
    with TestClient(m.app) as c:
        yield c


def test_si_el_modelo_no_carga_responde_la_regla(api_sin_modelo):
    s = api_sin_modelo.get("/salud").json()
    assert s["estado"] == "degradado" and not s["modelo_cargado"] and s["respaldo_activo"]
    atrasado = api_sin_modelo.post("/predecir", json={**cliente_valido(), "pay_status_m1": 2}).json()
    al_dia = api_sin_modelo.post("/predecir", json={**cliente_valido(), "pay_status_m1": 0}).json()
    for j in (atrasado, al_dia):
        assert j["respaldo"] is True and j["fuente"] == "regla_negocio"
        assert j["motivo_respaldo"] and j["version_modelo"] is None
    assert atrasado["decision"] == "contactar" and al_dia["decision"] == "no_contactar"
    assert api_sin_modelo.get("/modelo").json()["fuente_activa"] == "regla_negocio"


def test_interruptor_de_respaldo(api):
    assert api.post("/predecir", json=cliente_valido()).json()["fuente"] == "modelo"
    r = api.post("/admin/respaldo", json={"activo": True, "motivo": "prueba de retirada"}, headers=TOKEN)
    assert r.status_code == 200 and r.json()["respaldo_activo"]
    j = api.post("/predecir", json=cliente_valido()).json()
    assert j["respaldo"] and j["fuente"] == "regla_negocio" and j["motivo_respaldo"] == "prueba de retirada"
    api.post("/admin/respaldo", json={"activo": False}, headers=TOKEN)
    assert api.post("/predecir", json=cliente_valido()).json()["fuente"] == "modelo"


def test_modo_respaldo_por_variable_de_entorno(tmp_path, monkeypatch):
    import api.main as m

    monkeypatch.setenv("MODO_RESPALDO", "1")
    monkeypatch.setattr(m, "REGISTRO_DB", tmp_path / "registro.db")
    monkeypatch.setattr(m, "ESTADO", m.EstadoModelo())
    with TestClient(m.app) as c:
        j = c.post("/predecir", json=cliente_valido()).json()
        assert j["respaldo"] and j["fuente"] == "regla_negocio"
        assert c.get("/salud").json()["modelo_cargado"]      # el modelo cargó, pero el interruptor manda

"""Contrato de la API: entradas válidas -> 200 con el esquema esperado; inválidas -> 422 con el campo."""
from mlflow.models import Model

from api.esquemas import FEATURES
from src.contrato import FEATURES as FEATURES_CONTRATO
from tests.conftest import FIXTURES, cliente_valido

CAMPOS_RESPUESTA = {"id_prediccion", "customer_id", "probabilidad", "decision", "accion", "umbral",
                    "version_modelo", "alias", "run_id", "fuente", "respaldo", "motivo_respaldo",
                    "latencia_ms", "timestamp"}


def test_esquema_coincide_con_firma_mlflow():
    firma = Model.load(str(FIXTURES / "modelo")).signature
    assert firma is not None, "el modelo exportado debe traer su firma"
    assert [c.name for c in firma.inputs.inputs] == FEATURES == FEATURES_CONTRATO


def test_entrada_valida_devuelve_200_y_esquema(api, referencia):
    r = api.post("/predecir", json=cliente_valido())
    assert r.status_code == 200
    j = r.json()
    assert set(j) == CAMPOS_RESPUESTA
    assert 0.0 <= j["probabilidad"] <= 1.0
    assert j["decision"] in {"contactar", "no_contactar"}
    assert j["accion"]
    assert j["version_modelo"] == referencia["version"]
    assert j["run_id"] == referencia["run_id"]
    assert j["fuente"] == "modelo" and j["respaldo"] is False


def test_clientes_reales_de_la_muestra(api, muestra):
    for _, fila in muestra.head(20).iterrows():
        assert api.post("/predecir", json=cliente_valido(fila)).status_code == 200


def _campos_con_error(r):
    return {e["campo"] for e in r.json()["detail"]}


def test_campo_faltante_devuelve_422(api):
    d = cliente_valido()
    d.pop("pay_status_m1")
    r = api.post("/predecir", json=d)
    assert r.status_code == 422
    assert "pay_status_m1" in _campos_con_error(r)


def test_tipo_incorrecto_devuelve_422(api):
    r = api.post("/predecir", json={**cliente_valido(), "age": "treinta y cinco"})
    assert r.status_code == 422
    assert "age" in _campos_con_error(r)


def test_categoria_desconocida_devuelve_422(api):
    r = api.post("/predecir", json={**cliente_valido(), "education_code": 9})
    assert r.status_code == 422
    assert "education_code" in _campos_con_error(r)


def test_fuera_de_rango_devuelve_422(api):
    r = api.post("/predecir", json={**cliente_valido(), "pay_status_m1": 15})
    assert r.status_code == 422
    assert "pay_status_m1" in _campos_con_error(r)


def test_campo_extra_devuelve_422(api):
    r = api.post("/predecir", json={**cliente_valido(), "sexo": 2})
    assert r.status_code == 422
    assert "sexo" in _campos_con_error(r)


def test_salud_y_modelo(api, referencia):
    s = api.get("/salud").json()
    assert s["estado"] == "ok" and s["modelo_cargado"] and not s["respaldo_activo"]
    m = api.get("/modelo").json()
    assert m["version"] == referencia["version"] and m["run_id"] == referencia["run_id"]
    assert m["fuente_activa"] == "modelo"


def test_cada_prediccion_queda_registrada(api):
    api.post("/predecir", json=cliente_valido(), headers={"X-Lote": "3"})
    api.post("/predecir", json={**cliente_valido(), "age": 5}, headers={"X-Lote": "3"})
    tok = {"X-Token": "token-de-prueba"}
    p = api.get("/admin/pendientes", params={"tabla": "predicciones"}, headers=tok).json()
    assert p["n"] == 1
    fila = p["filas"][0]
    for campo in ("prediccion_id", "ts", "entradas", "probabilidad", "decision", "version_modelo", "latencia_ms"):
        assert fila[campo] is not None
    assert fila["lote_id"] == 3
    s = api.get("/admin/pendientes", params={"tabla": "solicitudes"}, headers=tok).json()
    assert sorted(f["status"] for f in s["filas"] if f["ruta"] == "/predecir") == [200, 422]


def test_admin_exige_token(api):
    assert api.post("/admin/respaldo", json={"activo": True}).status_code == 401
    assert api.get("/admin/pendientes", headers={"X-Token": "otro"}).status_code == 401

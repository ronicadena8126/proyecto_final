"""
Servicio de predicción "Anticipar la mora" (FastAPI).

  GET  /salud      estado del servicio y si el modelo cargó
  GET  /modelo     nombre, versión, alias y run_id del modelo en uso
  POST /predecir   probabilidad, decisión, acción, versión del modelo e id de la predicción

El modelo se carga desde MODELO_URI (por defecto models:/anticipar-mora-modelo@champion). Un hilo
revisa el alias cada RECARGA_SEGUNDOS: si alguien mueve @champion (promoción o rollback), la API
carga la nueva versión sin reconstruir la imagen ni reiniciar el contenedor.

Respaldo: si el modelo no carga, o si se activa el interruptor (MODO_RESPALDO=1 al arrancar, o
POST /admin/respaldo con el token de administración), la API responde con la regla de negocio del
PSet 3 (contactar si hay >= 1 mes de atraso) e indica respaldo=true en cada respuesta.

Cada predicción y cada solicitud quedan en un registro local (SQLite en un volumen). Kestra las
descarga con GET /admin/pendientes y las carga en Snowflake (tablas PREDICCIONES y SOLICITUDES);
así la API no guarda credenciales del almacén y una caída de Snowflake no tumba el servicio.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from api.esquemas import FEATURES, CambioRespaldo, Cliente, InfoModelo, Prediccion, Salud  # noqa: E402
from src import negocio  # noqa: E402

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

NOMBRE_MODELO = os.environ.get("MODELO_NOMBRE", "anticipar-mora-modelo")
ALIAS = os.environ.get("MODELO_ALIAS", "champion")
MODELO_URI = os.environ.get("MODELO_URI", f"models:/{NOMBRE_MODELO}@{ALIAS}")
RECARGA_SEGUNDOS = float(os.environ.get("RECARGA_SEGUNDOS", "30"))
REGISTRO_DB = Path(os.environ.get("REGISTRO_DB", RAIZ / "data" / "registro_api.db"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _verdadero(v: str | None) -> bool:
    return str(v).strip().lower() in {"1", "true", "si", "sí", "yes", "on"}


# ----------------------------------------------------------------------------- estado del modelo
class EstadoModelo:
    def __init__(self):
        self.modelo = None
        self.info: dict = {}
        self.cargado_en: str | None = None
        self.error: str | None = None
        self.interruptor = _verdadero(os.environ.get("MODO_RESPALDO"))
        self.motivo_interruptor = "MODO_RESPALDO=1 al arrancar" if self.interruptor else None
        self.lock = threading.Lock()

    # Respaldo activo = el interruptor está encendido o no hay modelo cargado.
    @property
    def respaldo(self) -> bool:
        return self.interruptor or self.modelo is None

    @property
    def motivo_respaldo(self) -> str | None:
        if self.interruptor:
            return self.motivo_interruptor
        if self.modelo is None:
            return f"modelo no disponible: {self.error}" if self.error else "modelo no disponible"
        return None

    @property
    def umbral(self) -> float:
        if self.respaldo:
            return 0.5
        return max(self.info.get("corte_efectivo", 0.34), self.info.get("umbral_decision", 0.05))

    def _info_local(self, ruta: Path) -> dict:
        """Modelo exportado con save_model (CI): los metadatos vienen en referencia.json."""
        ref = ruta.parent / "referencia.json"
        meta = json.loads(ref.read_text()) if ref.exists() else {}
        return {"nombre": NOMBRE_MODELO, "alias": meta.get("alias", "fixture"),
                "version": meta.get("version"), "run_id": meta.get("run_id"),
                "corte_efectivo": meta.get("corte_efectivo", 0.34),
                "umbral_decision": meta.get("umbral_decision", 0.05),
                "datos_hash": meta.get("datos_hash"), "datos_tabla_origen": meta.get("datos_tabla_origen")}

    def cargar(self) -> None:
        import mlflow

        try:
            ruta = Path(MODELO_URI)
            if ruta.exists():
                info = self._info_local(ruta)
                modelo = mlflow.sklearn.load_model(str(ruta))
            else:
                from src.registro import info_version

                info = info_version(ALIAS)
                modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}/{info['version']}")
            with self.lock:
                self.modelo, self.info, self.error = modelo, info, None
                self.cargado_en = ahora()
            log.info("Modelo cargado: %s v%s (run %s)", NOMBRE_MODELO, info.get("version"), info.get("run_id"))
        except Exception as e:  # el servicio sigue arriba con el respaldo
            with self.lock:
                self.error = f"{type(e).__name__}: {e}"[:300]
                if self.modelo is None:
                    self.info = {}
            log.error("No se pudo cargar el modelo (%s). Se usa el respaldo.", self.error)

    def version_registrada(self) -> int | None:
        from src.registro import version_por_alias

        return int(version_por_alias(ALIAS).version)

    def vigilar(self, parar: threading.Event) -> None:
        """Recarga si el alias apunta a otra versión (promoción o rollback) o si el modelo no cargó."""
        while not parar.wait(RECARGA_SEGUNDOS):
            if Path(MODELO_URI).exists():
                continue
            try:
                v = self.version_registrada()
                if self.modelo is None or v != self.info.get("version"):
                    log.info("@%s apunta a v%s (en uso: v%s): recargando", ALIAS, v, self.info.get("version"))
                    self.cargar()
            except Exception as e:
                log.warning("No se pudo consultar el registro: %s", e)


ESTADO = EstadoModelo()


# ----------------------------------------------------------------------------- registro local
def _db() -> sqlite3.Connection:
    REGISTRO_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(REGISTRO_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def iniciar_registro() -> None:
    with _db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT, prediccion_id TEXT, ts TEXT, lote_id INTEGER,
            customer_id INTEGER, entradas TEXT, probabilidad REAL, decision TEXT, umbral REAL,
            fuente TEXT, respaldo INTEGER, version_modelo INTEGER, run_id TEXT, latencia_ms REAL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS solicitudes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, lote_id INTEGER, ruta TEXT, metodo TEXT,
            status INTEGER, latencia_ms REAL, detalle TEXT)""")


def guardar_prediccion(fila: dict) -> None:
    with _db() as con:
        con.execute("""INSERT INTO predicciones (prediccion_id, ts, lote_id, customer_id, entradas,
            probabilidad, decision, umbral, fuente, respaldo, version_modelo, run_id, latencia_ms)
            VALUES (:prediccion_id, :ts, :lote_id, :customer_id, :entradas, :probabilidad, :decision,
            :umbral, :fuente, :respaldo, :version_modelo, :run_id, :latencia_ms)""", fila)


def guardar_solicitud(fila: dict) -> None:
    try:
        with _db() as con:
            con.execute("""INSERT INTO solicitudes (ts, lote_id, ruta, metodo, status, latencia_ms, detalle)
                VALUES (:ts, :lote_id, :ruta, :metodo, :status, :latencia_ms, :detalle)""", fila)
    except Exception as e:  # registrar nunca debe tumbar una respuesta
        log.warning("No se pudo registrar la solicitud: %s", e)


# ----------------------------------------------------------------------------- aplicación
@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    iniciar_registro()
    ESTADO.cargar()
    parar = threading.Event()
    hilo = threading.Thread(target=ESTADO.vigilar, args=(parar,), daemon=True)
    hilo.start()
    yield
    parar.set()


app = FastAPI(title="Anticipar la mora — servicio de predicción", version="1.0.0", lifespan=ciclo_de_vida)


def _lote(request: Request) -> int | None:
    v = request.headers.get("X-Lote")
    return int(v) if v and v.lstrip("-").isdigit() else None


@app.middleware("http")
async def medir(request: Request, call_next):
    request.state.t0 = time.perf_counter()
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception:
        status = 500
        resp = JSONResponse({"detail": "error interno"}, status_code=500)
    if request.url.path in ("/predecir", "/salud", "/modelo"):
        guardar_solicitud({"ts": ahora(), "lote_id": _lote(request), "ruta": request.url.path,
                           "metodo": request.method, "status": status,
                           "latencia_ms": (time.perf_counter() - request.state.t0) * 1000,
                           "detalle": getattr(request.state, "detalle", None)})
    return resp


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError):
    """422 con el campo que falló y el motivo, en una forma corta y legible."""
    errores = [{"campo": ".".join(str(p) for p in e["loc"] if p != "body"), "error": e["msg"],
                "tipo": e["type"]} for e in exc.errors()]
    request.state.detalle = ";".join(f"{e['campo']}:{e['tipo']}" for e in errores)[:500]
    return JSONResponse(status_code=422, content={"detail": errores})


def exigir_token(x_token: str | None = Header(default=None)):
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Administración deshabilitada: falta ADMIN_TOKEN")
    if x_token != ADMIN_TOKEN:
        raise HTTPException(401, "Token de administración inválido")


@app.get("/salud", response_model=Salud)
def salud():
    return Salud(estado="ok" if not ESTADO.respaldo else "degradado",
                 modelo_cargado=ESTADO.modelo is not None, respaldo_activo=ESTADO.respaldo,
                 motivo_respaldo=ESTADO.motivo_respaldo, version_modelo=ESTADO.info.get("version"),
                 cargado_en=ESTADO.cargado_en)


@app.get("/modelo", response_model=InfoModelo)
def modelo():
    i = ESTADO.info
    return InfoModelo(nombre=NOMBRE_MODELO, version=i.get("version"), alias=i.get("alias"),
                      run_id=i.get("run_id"), uri=MODELO_URI, umbral=ESTADO.umbral,
                      fuente_activa="regla_negocio" if ESTADO.respaldo else "modelo",
                      cargado_en=ESTADO.cargado_en, datos_hash=i.get("datos_hash"),
                      datos_tabla_origen=i.get("datos_tabla_origen"))


@app.post("/predecir", response_model=Prediccion)
def predecir(cliente: Cliente, request: Request):
    entrada = cliente.model_dump()
    with ESTADO.lock:
        modelo, info, respaldo = ESTADO.modelo, dict(ESTADO.info), ESTADO.respaldo
        motivo, umbral = ESTADO.motivo_respaldo, ESTADO.umbral
    if respaldo:
        p, decision = negocio.decision_regla(entrada["pay_status_m1"])
        fuente = "regla_negocio"
    else:
        X = pd.DataFrame([{c: entrada[c] for c in FEATURES}]).astype(float)
        p = float(modelo.predict_proba(X)[0, 1])
        decision = negocio.decision_modelo(p, umbral)
        fuente = "modelo"
    lat = (time.perf_counter() - request.state.t0) * 1000
    pid, ts = str(uuid.uuid4()), ahora()
    version = None if respaldo else info.get("version")
    run_id = None if respaldo else info.get("run_id")
    guardar_prediccion({"prediccion_id": pid, "ts": ts, "lote_id": _lote(request),
                        "customer_id": entrada.get("customer_id"), "entradas": json.dumps(entrada),
                        "probabilidad": p, "decision": decision, "umbral": umbral, "fuente": fuente,
                        "respaldo": int(respaldo), "version_modelo": version, "run_id": run_id,
                        "latencia_ms": lat})
    return Prediccion(id_prediccion=pid, customer_id=entrada.get("customer_id"), probabilidad=round(p, 6),
                      decision=decision, accion=negocio.ACCIONES[decision], umbral=round(umbral, 4),
                      version_modelo=version, alias=None if respaldo else info.get("alias"), run_id=run_id,
                      fuente=fuente, respaldo=respaldo, motivo_respaldo=motivo, latencia_ms=round(lat, 3),
                      timestamp=ts)


# ----------------------------------------------------------------------------- administración
@app.post("/admin/respaldo", dependencies=[Depends(exigir_token)])
def interruptor(cambio: CambioRespaldo):
    """Interruptor de respaldo (sección 8.4): fuerza la regla de negocio sin tocar el modelo."""
    with ESTADO.lock:
        ESTADO.interruptor = cambio.activo
        ESTADO.motivo_interruptor = cambio.motivo if cambio.activo else None
    log.warning("Interruptor de respaldo -> %s (%s)", cambio.activo, cambio.motivo)
    return salud()


@app.post("/admin/recargar", dependencies=[Depends(exigir_token)])
def recargar():
    ESTADO.cargar()
    return modelo()


@app.get("/admin/pendientes", dependencies=[Depends(exigir_token)])
def pendientes(tabla: str = "predicciones", desde_id: int = 0, limite: int = 20000):
    if tabla not in ("predicciones", "solicitudes"):
        raise HTTPException(400, "tabla debe ser predicciones o solicitudes")
    with _db() as con:
        df = pd.read_sql_query(f"SELECT * FROM {tabla} WHERE id > ? ORDER BY id LIMIT ?", con,
                               params=(desde_id, limite))
        ultimo = con.execute(f"SELECT COALESCE(MAX(id), 0) FROM {tabla}").fetchone()[0]
    return {"tabla": tabla, "n": len(df), "ultimo_id": int(ultimo),
            "filas": json.loads(df.to_json(orient="records"))}

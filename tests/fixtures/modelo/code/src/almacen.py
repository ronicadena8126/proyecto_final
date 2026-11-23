"""
Acceso a las tablas de operación (PREDICCIONES, MONITOREO, ALERTAS, ...).

Producción usa Snowflake (base PSET2_MORA, esquema PRODUCCION, mismo usuario key-pair del PSet 2).
Las pruebas y el modo local usan un archivo DuckDB con el mismo esquema y las mismas tablas,
para que el código de simulación, monitoreo y reentrenamiento sea idéntico en ambos casos.

    ALMACEN=snowflake   (por defecto)  -> Snowflake
    ALMACEN=local                      -> data/almacen.duckdb (o ALMACEN_RUTA)

Credenciales de Snowflake (nunca en el repositorio):
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE
    y la llave privada como archivo (SNOWFLAKE_PRIVATE_KEY_PATH) o como texto PEM
    (SNOWFLAKE_PRIVATE_KEY, así la entregan los secretos de Kestra).
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ESQUEMA = "PRODUCCION"


def _cargar_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass


def modo() -> str:
    _cargar_env()
    return os.environ.get("ALMACEN", "snowflake").lower()


# ----------------------------------------------------------------------------- Snowflake
def _ruta_llave() -> str:
    pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "").strip()
    if pem:
        if "BEGIN" not in pem:          # los secretos de Kestra pueden llegar en base64
            import base64

            pem = base64.b64decode(pem).decode()
        f = Path(tempfile.gettempdir()) / "sf_key.p8"
        f.write_text(pem if pem.endswith("\n") else pem + "\n")
        f.chmod(0o600)
        return str(f)
    return os.path.expanduser(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])


_CONEXIONES: dict = {}


def conexion_snowflake(esquema: str | None = ESQUEMA):
    """Una conexión por proceso y esquema (abrir una conexión cuesta 1-2 s)."""
    con = _CONEXIONES.get(esquema)
    if con is not None and not con.is_closed():
        return con
    con = _conectar(esquema)
    _CONEXIONES[esquema] = con
    return con


def _conectar(esquema: str | None):
    import snowflake.connector

    _cargar_env()
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        private_key_file=_ruta_llave(),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=esquema,
    )


# ----------------------------------------------------------------------------- DuckDB local
def ruta_local() -> Path:
    return Path(os.environ.get("ALMACEN_RUTA", RAIZ / "data" / "almacen.duckdb"))


@contextmanager
def _duck(read_only: bool = False):
    import duckdb

    ruta = ruta_local()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ruta), read_only=read_only and ruta.exists())
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {ESQUEMA}")
        yield con
    finally:
        con.close()


# ----------------------------------------------------------------------------- API común
def preparar() -> None:
    """Crea el esquema PRODUCCION si no existe (idempotente)."""
    if modo() == "local":
        with _duck():
            pass
        return
    conexion_snowflake(esquema=None).cursor().execute(f"CREATE SCHEMA IF NOT EXISTS {ESQUEMA}")


def leer(sql: str) -> pd.DataFrame:
    """SELECT -> DataFrame con columnas en minúscula. Usa `produccion.<tabla>` en el SQL."""
    if modo() == "local":
        with _duck() as con:
            df = con.execute(sql).df()
    else:
        df = conexion_snowflake().cursor().execute(sql).fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    return df


def ejecutar(sql: str) -> None:
    if modo() == "local":
        with _duck() as con:
            con.execute(sql)
        return
    conexion_snowflake().cursor().execute(sql)


def existe(tabla: str) -> bool:
    try:
        leer(f"SELECT 1 FROM {ESQUEMA}.{tabla} LIMIT 1")
        return True
    except Exception:
        return False


def escribir(tabla: str, df: pd.DataFrame, reemplazar: bool = False,
             borrar_donde: str | None = None) -> int:
    """Inserta df en PRODUCCION.<tabla> (la crea si no existe).

    reemplazar=True sobreescribe la tabla; borrar_donde="lote_id = 3" borra antes esas filas,
    así reprocesar un lote nunca duplica datos (idempotencia, igual que la ingesta del PSet 2).
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for c in df.columns:  # tipos simples y portables: fechas como texto ISO, bool como int
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
    if borrar_donde and existe(tabla) and not reemplazar:
        ejecutar(f"DELETE FROM {ESQUEMA}.{tabla} WHERE {borrar_donde}")
    if df.empty:
        return 0
    if modo() == "local":
        with _duck() as con:
            con.register("df_tmp", df)
            existe_t = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = ? AND lower(table_name) = ?",
                [ESQUEMA, tabla.lower()],
            ).fetchone()[0]
            if reemplazar or not existe_t:
                con.execute(f"CREATE OR REPLACE TABLE {ESQUEMA}.{tabla} AS SELECT * FROM df_tmp")
            else:
                cols = ", ".join(df.columns)
                con.execute(f"INSERT INTO {ESQUEMA}.{tabla} ({cols}) SELECT {cols} FROM df_tmp")
        return len(df)

    from snowflake.connector.pandas_tools import write_pandas

    df.columns = [c.upper() for c in df.columns]
    ok, _, n, _ = write_pandas(conexion_snowflake(), df, table_name=tabla.upper(), schema=ESQUEMA,
                               auto_create_table=True, overwrite=reemplazar, quote_identifiers=False)
    if not ok:
        raise RuntimeError(f"write_pandas falló en {tabla}")
    return int(n)

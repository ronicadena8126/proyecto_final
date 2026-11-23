"""
Carga de la OBT del proyecto "Anticipar la mora" y división train/test.

Fuente: PSET2_MORA.OBT.OBT_CUSTOMER_DEFAULT en Snowflake (construida en el PSet 2).
La primera lectura guarda una copia en data/obt.parquet (no se versiona); así los notebooks
siguientes usan exactamente el mismo snapshot aunque la tabla cambie después.

Variables de entorno (.env en la raíz de pset_3/):
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_PATH, SNOWFLAKE_ROLE,
  SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RAIZ = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OBT_CACHE", RAIZ / "data" / "obt.parquet"))
TABLA_ORIGEN = "PSET2_MORA.OBT.OBT_CUSTOMER_DEFAULT"
TARGET = "default_next_month"
ID = "customer_id"
SEMILLA = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Columnas que NO son features: identificadores, metadatos de carga, constantes,
# duplicados de otra columna y el propio target.
NO_FEATURES = [
    "customer_key", ID, "attributes_as_of", "observation_month_key", "label_month_key",
    "n_months_history",   # constante (= 6 para todos)
    "education",          # texto duplicado de education_code
    "age_band",           # derivada de age
    TARGET,
]


def _cargar_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass


def leer_snowflake() -> pd.DataFrame:
    from src.almacen import conexion_snowflake

    _cargar_env()
    conn = conexion_snowflake(esquema=None)   # llave como archivo o como texto (secretos de Kestra)
    df = conn.cursor().execute(f"SELECT * FROM {TABLA_ORIGEN}").fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    return df


def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos estables entre Snowflake y parquet, y orden determinista (por customer_id)."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == bool or str(df[c].dtype) == "boolean":
            df[c] = df[c].astype(int)
        elif df[c].dtype == object and c not in ("customer_key", "education", "age_band", "attributes_as_of"):
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype(float)
    df[TARGET] = df[TARGET].astype(int)
    df[ID] = df[ID].astype(int)
    return df.sort_values(ID).reset_index(drop=True)


def cargar_obt(refrescar: bool = False) -> tuple[pd.DataFrame, dict]:
    """Devuelve (obt, metadatos). Usa data/obt.parquet si existe, salvo refrescar=True."""
    if refrescar or not CACHE.exists():
        df = _normalizar(leer_snowflake())
        CACHE.parent.mkdir(exist_ok=True)
        df.to_parquet(CACHE, index=False)
        extraido = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (CACHE.with_suffix(".extraido")).write_text(extraido)
    df = _normalizar(pd.read_parquet(CACHE))
    extraido_f = CACHE.with_suffix(".extraido")
    meta = {
        "tabla_origen": TABLA_ORIGEN,
        "fecha_corte_snapshot": "2005-09-30",       # último estado de cuenta observado
        "fecha_extraccion": extraido_f.read_text() if extraido_f.exists() else "desconocida",
        "hash_dataset": hash_dataset(df),
        "n_filas": len(df),
        "n_columnas": df.shape[1],
    }
    return df, meta


def hash_dataset(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df.sort_values(ID).reset_index(drop=True), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def columnas_vacias(df: pd.DataFrame) -> list[str]:
    """Columnas 100% nulas en el snapshot (no aportan información y rompen el imputador)."""
    return [c for c in df.columns if df[c].isna().all()]


def columnas_features(df: pd.DataFrame) -> list[str]:
    vacias = set(columnas_vacias(df))
    return [c for c in df.columns if c not in NO_FEATURES and c not in vacias]


def dividir(df: pd.DataFrame):
    """Hold-out estratificado 80/20, una sola vez y con semilla fija."""
    X = df[columnas_features(df)]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEMILLA)


def particion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Las mismas filas de entrenamiento y prueba del PSet 3, con id y target (para la simulación)."""
    X_tr, X_te, _, _ = dividir(df)
    return df.loc[X_tr.index].copy(), df.loc[X_te.index].copy()


def esquema_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)

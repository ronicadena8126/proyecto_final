"""
Esquemas de entrada y salida de la API.

La entrada replica la firma registrada en MLflow para el champion (43 columnas de la OBT, en el mismo
orden; tests/test_contrato.py verifica que nombres y orden coincidan con la firma del modelo).
Los rangos son los del dominio (diccionario de datos UCI y reglas de Silver del PSet 2), con margen
para valores legítimos que aún no aparecieron; las categorías son las que vio el entrenamiento.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EstadoPago = Field(ge=-2, le=8, description="-2 sin consumo, -1 pagó el total, 0 pagó el mínimo, k = k meses de atraso")
Monto = Field(ge=-5_000_000, le=5_000_000, description="NT$")
Pago = Field(ge=0, le=5_000_000, description="NT$")
Util = Field(default=..., ge=-50, le=50, description="saldo / cupo (puede ser nulo si el cupo es 0)")
Binaria = Literal[0, 1]


class Cliente(BaseModel):
    """Un cliente al cierre de su último estado de cuenta (una fila de la OBT)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "customer_id": 12345, "limit_bal": 120000, "education_code": 2, "is_education_imputed": 0,
        "is_marriage_imputed": 0, "age": 35, "is_inactive_6m": 0, "months_with_credit_balance": 0,
        "pay_status_m1": 2, "bill_amt_m1": 98000, "pay_amt_m1": 0, "util_m1": 0.82,
        "pay_status_m2": 2, "bill_amt_m2": 96000, "pay_amt_m2": 3000, "util_m2": 0.80,
        "pay_status_m3": 0, "bill_amt_m3": 93000, "pay_amt_m3": 3500, "util_m3": 0.78,
        "pay_status_m4": 0, "bill_amt_m4": 90000, "pay_amt_m4": 3200, "util_m4": 0.75,
        "pay_status_m5": 0, "bill_amt_m5": 88000, "pay_amt_m5": 3100, "util_m5": 0.73,
        "pay_status_m6": 0, "bill_amt_m6": 85000, "pay_amt_m6": 3000, "util_m6": 0.71,
        "max_delay_6m": 2, "n_delinquent_months_6m": 2, "n_delinquent_months_3m": 2,
        "avg_utilization_6m": 0.77, "max_utilization_6m": 0.82, "total_pay_amt_6m": 15800,
        "avg_bill_amt_6m": 91667, "n_overlimit_months_6m": 0, "n_credit_balance_months_6m": 0,
        "n_inconsistent_months_6m": 0, "bill_trend_m1_vs_m6": 13000, "is_delinquent_last_month": 1}})

    customer_id: Optional[int] = Field(default=None, ge=1, description="Solo para trazabilidad; el modelo no lo usa")
    limit_bal: float = Field(gt=0, le=5_000_000, description="Cupo de la tarjeta, NT$")
    education_code: Literal[1, 2, 3, 4] = Field(description="1 posgrado, 2 universidad, 3 secundaria, 4 otros")
    is_education_imputed: Binaria
    is_marriage_imputed: Binaria
    age: int = Field(ge=18, le=100)
    is_inactive_6m: Binaria
    months_with_credit_balance: int = Field(ge=0, le=6)
    pay_status_m1: int = EstadoPago
    bill_amt_m1: float = Monto
    pay_amt_m1: float = Pago
    util_m1: Optional[float] = Util
    pay_status_m2: int = EstadoPago
    bill_amt_m2: float = Monto
    pay_amt_m2: float = Pago
    util_m2: Optional[float] = Util
    pay_status_m3: int = EstadoPago
    bill_amt_m3: float = Monto
    pay_amt_m3: float = Pago
    util_m3: Optional[float] = Util
    pay_status_m4: int = EstadoPago
    bill_amt_m4: float = Monto
    pay_amt_m4: float = Pago
    util_m4: Optional[float] = Util
    pay_status_m5: int = EstadoPago
    bill_amt_m5: float = Monto
    pay_amt_m5: float = Pago
    util_m5: Optional[float] = Util
    pay_status_m6: int = EstadoPago
    bill_amt_m6: float = Monto
    pay_amt_m6: float = Pago
    util_m6: Optional[float] = Util
    max_delay_6m: int = Field(ge=0, le=8)
    n_delinquent_months_6m: int = Field(ge=0, le=6)
    n_delinquent_months_3m: int = Field(ge=0, le=3)
    avg_utilization_6m: Optional[float] = Util
    max_utilization_6m: Optional[float] = Util
    total_pay_amt_6m: float = Field(ge=0, le=30_000_000)
    avg_bill_amt_6m: float = Monto
    n_overlimit_months_6m: int = Field(ge=0, le=6)
    n_credit_balance_months_6m: int = Field(ge=0, le=6)
    n_inconsistent_months_6m: int = Field(ge=0, le=6)
    bill_trend_m1_vs_m6: float = Field(ge=-10_000_000, le=10_000_000)
    is_delinquent_last_month: Binaria


FEATURES = [c for c in Cliente.model_fields if c != "customer_id"]


class Prediccion(BaseModel):
    id_prediccion: str
    customer_id: Optional[int]
    probabilidad: float = Field(description="Probabilidad de impago el próximo mes")
    decision: Literal["contactar", "no_contactar"]
    accion: str
    umbral: float
    version_modelo: Optional[int]
    alias: Optional[str]
    run_id: Optional[str]
    fuente: Literal["modelo", "regla_negocio"]
    respaldo: bool
    motivo_respaldo: Optional[str] = None
    latencia_ms: float
    timestamp: str


class Salud(BaseModel):
    estado: Literal["ok", "degradado"]
    modelo_cargado: bool
    respaldo_activo: bool
    motivo_respaldo: Optional[str]
    version_modelo: Optional[int]
    cargado_en: Optional[str]


class InfoModelo(BaseModel):
    nombre: str
    version: Optional[int]
    alias: Optional[str]
    run_id: Optional[str]
    uri: str
    umbral: float
    fuente_activa: Literal["modelo", "regla_negocio"]
    cargado_en: Optional[str]
    datos_hash: Optional[str] = None
    datos_tabla_origen: Optional[str] = None


class CambioRespaldo(BaseModel):
    activo: bool
    motivo: str = "interruptor manual"

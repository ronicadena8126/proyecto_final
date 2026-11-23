"""
Contrato de entrada del modelo: las 43 columnas de la OBT que recibe el pipeline, en el orden de la
firma registrada en MLflow. Lo usan la simulación, el monitoreo y el reentrenamiento; la API valida
lo mismo con Pydantic (api/esquemas.py) y tests/test_contrato.py verifica que las tres coincidan.

Se fija aquí (y no se deduce de la OBT) para que un cambio en la tabla, por ejemplo una columna nueva o
avg_payment_ratio_6m con datos otra vez, no cambie en silencio lo que recibe el modelo en producción.
"""
FEATURES = [
    "limit_bal",
    "education_code",
    "is_education_imputed",
    "is_marriage_imputed",
    "age",
    "is_inactive_6m",
    "months_with_credit_balance",
    "pay_status_m1",
    "bill_amt_m1",
    "pay_amt_m1",
    "util_m1",
    "pay_status_m2",
    "bill_amt_m2",
    "pay_amt_m2",
    "util_m2",
    "pay_status_m3",
    "bill_amt_m3",
    "pay_amt_m3",
    "util_m3",
    "pay_status_m4",
    "bill_amt_m4",
    "pay_amt_m4",
    "util_m4",
    "pay_status_m5",
    "bill_amt_m5",
    "pay_amt_m5",
    "util_m5",
    "pay_status_m6",
    "bill_amt_m6",
    "pay_amt_m6",
    "util_m6",
    "max_delay_6m",
    "n_delinquent_months_6m",
    "n_delinquent_months_3m",
    "avg_utilization_6m",
    "max_utilization_6m",
    "total_pay_amt_6m",
    "avg_bill_amt_6m",
    "n_overlimit_months_6m",
    "n_credit_balance_months_6m",
    "n_inconsistent_months_6m",
    "bill_trend_m1_vs_m6",
    "is_delinquent_last_month",
]

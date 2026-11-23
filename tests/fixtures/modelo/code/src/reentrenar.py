"""
Reentrenamiento (sección 7.2). Lo orquesta kestra/reentrenamiento.yml:

  python -m src.reentrenar entrenar --lote 8 --motivo desempeno
  python -m src.reentrenar decidir  --lote 8 --version 3 --aprobar true --comentario "..."

entrenar
  1. Ventana: TODO el histórico (entrenamiento del PSet 3 + lotes de producción con target), con
     mayor peso en los lotes recientes: peso(lote k) = max(1, W * 0,5^((k_ult - k) / h)), W = 10,
     h = 1 lote. Un cambio de régimen se ve primero en los lotes recientes; sin pesos, 600 clientes
     nuevos quedan diluidos entre 24.000 antiguos. Con W = 10 el último lote pesa como ~6.000 filas
     (~20% de la masa), sin olvidar la historia.
  2. Solo lotes cuyo target ya llegó. El más reciente se RESERVA para comparar champion y challenger
     (ninguno de los dos lo vio). Los registros rechazados por la API (corruptos) no se usan.
  3. Misma preparación (src/pipeline.py) e hiperparámetros del champion: clone() del pipeline
     registrado. No se repite el tuning.
  4. Corrida en MLflow con fase=reentrenamiento y el motivo; nueva versión con alias @challenger.
  5. Regla de promoción (sección 4), fijada antes de ver el resultado:
       ahorro(challenger) - ahorro(champion) > sigma_CV(ahorro)   en el lote reservado, y
       AUC(challenger) >= AUC(champion) - cv_std                 (no empeora la métrica técnica)
     sigma_CV(ahorro) = desviación del ahorro entre los 5 pliegues de la CV del challenger,
     escalada al tamaño del lote. El ahorro se mide a igual capacidad (top 20% del lote).

decidir
  Después de la pausa de Kestra: si la persona aprueba, @champion pasa a la nueva versión
  (y el anterior queda en @anterior para rollback); si rechaza, el champion queda intacto.
  En ambos casos se registra la decisión y el motivo (MLflow + tabla REENTRENAMIENTOS).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import almacen, negocio, registro
from src.contrato import FEATURES
from src.datos import ID, SEMILLA, TARGET, cargar_obt, particion
from src.tracking import MODELO_REGISTRO, commit_git, configurar, log_datos, log_modelo

PESO_MAX = 10.0
VIDA_MEDIA = 1.0
API_URL = os.environ.get("API_URL", "")


def _out(**kv):
    print("::" + json.dumps({"outputs": kv}, default=str) + "::")


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pesos_lotes(lotes: pd.Series, k_ult: int) -> np.ndarray:
    return np.maximum(1.0, PESO_MAX * 0.5 ** ((k_ult - lotes.to_numpy()) / VIDA_MEDIA))


def ventana(lote: int):
    """(X, y, pesos, origen) de entrenamiento y (X_eval, y_eval) del lote reservado."""
    df, meta = cargar_obt()
    train, _ = particion(df)
    feats = FEATURES
    t = almacen.leer(f"""SELECT {ID} AS customer_id, lote_id, {TARGET} AS target FROM {almacen.ESQUEMA}.TARGETS
                         WHERE disponible_en_lote <= {int(lote)}""")
    etiquetados = sorted(t.lote_id.unique())
    if not etiquetados:
        raise RuntimeError("Todavía no llega ningún target de producción")
    k_eval = int(max(etiquetados))
    L = almacen.leer(f"""SELECT * FROM {almacen.ESQUEMA}.LOTES WHERE lote_id IN ({",".join(map(str, etiquetados))})
                         AND registro_corrupto = 0""")
    L = L.merge(t.rename(columns={"customer_id": ID}), on=[ID, "lote_id"])
    ev = L[L.lote_id == k_eval]
    hist = L[L.lote_id < k_eval]
    k_ult = int(hist.lote_id.max()) if len(hist) else 0
    X = pd.concat([train[feats], hist[feats]], ignore_index=True).astype(float)
    y = pd.concat([train[TARGET], hist["target"]], ignore_index=True).astype(int)
    w = np.r_[np.ones(len(train)), pesos_lotes(hist.lote_id, k_ult) if len(hist) else []]
    origen = pd.Series(["entrenamiento_pset3"] * len(train) + [f"lote_{k}" for k in hist.lote_id])
    return X, y, w, origen, ev[feats].astype(float), ev["target"].astype(int).to_numpy(), k_eval, meta, feats


def evaluar_politica(y, score, saldo) -> dict:
    """Ahorro a la capacidad operativa (top 20% del lote) y AUC."""
    contactar = negocio.contactar_top(score, 0.0)
    return {"auc": float(roc_auc_score(y, score)), "ahorro": float(negocio.ahorro(y, contactar, saldo)),
            "moras_detectadas": int((contactar & (y == 1)).sum()), "contactados": int(contactar.sum())}


def chequeo_comportamiento(modelo, X: pd.DataFrame, corte: float) -> dict:
    """Las mismas pruebas de comportamiento de tests/test_comportamiento.py, sobre el lote reservado:
    informan a la persona que aprueba (un modelo que aprende "más atraso = menos riesgo" no se promueve)."""
    al_dia = X[X["pay_status_m1"] <= 0]
    Y = al_dia.copy()
    Y["pay_status_m1"], Y["is_delinquent_last_month"] = 2, 1
    Y["max_delay_6m"] = np.maximum(Y["max_delay_6m"], 2)
    Y["n_delinquent_months_3m"] = np.minimum(3, Y["n_delinquent_months_3m"] + 1)
    Y["n_delinquent_months_6m"] = np.minimum(6, Y["n_delinquent_months_6m"] + 1)
    p0, p2 = modelo.predict_proba(al_dia)[:, 1], modelo.predict_proba(Y)[:, 1]
    con_saldo = X[X["bill_amt_m2"] > 0]
    Z = con_saldo.copy()
    extra = np.maximum(Z["bill_amt_m2"] - Z["pay_amt_m1"], 0)
    Z["pay_amt_m1"] += extra
    Z["total_pay_amt_6m"] += extra
    q0, q1 = modelo.predict_proba(con_saldo)[:, 1], modelo.predict_proba(Z)[:, 1]
    return {"viol_atraso": int((p2 < p0 - 0.01).sum()), "viol_pago": int(((q0 < corte) & (q1 >= corte)).sum())}


def cv_challenger(pipe, X, y, w, n_lote: int, saldo: float) -> dict:
    """CV de 5 pliegues con los mismos pesos: AUC media/desv., sigma del ahorro y corte OOF."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    oof = np.zeros(len(y))
    aucs, ahorros = [], []
    for tr, va in cv.split(X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = clone(pipe).fit(X.iloc[tr], y.iloc[tr], modelo__sample_weight=w[tr])
        s = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = s
        aucs.append(roc_auc_score(y.iloc[va], s))
        ahorros.append(negocio.ahorro(y.iloc[va], negocio.contactar_top(s, 0.0), saldo) * n_lote / len(va))
    return {"cv_roc_auc_mean": float(np.mean(aucs)), "cv_roc_auc_std": float(np.std(aucs)),
            "cv_ahorro_lote_mean": float(np.mean(ahorros)), "cv_ahorro_lote_std": float(np.std(ahorros)),
            "corte_efectivo": negocio.corte_capacidad(oof)}


def lote_actual() -> int:
    """Lote de producción más reciente (para el disparador por calendario, que no trae lote)."""
    d = almacen.leer(f"SELECT MAX(lote_id) AS l FROM {almacen.ESQUEMA}.PREDICCIONES")
    return int(d.l.iloc[0]) + 1


def entrenar(lote: int, motivo: str) -> dict:
    if lote <= 0:
        lote = lote_actual()
        print(f"Disparo sin lote (calendario): se usa el lote actual {lote}")
    configurar()
    c = registro.cliente()
    champ_info = registro.info_version("champion", c)
    champion = mlflow.sklearn.load_model(f"models:/{MODELO_REGISTRO}/{champ_info['version']}")
    X, y, w, origen, X_ev, y_ev, k_eval, meta, feats = ventana(lote)
    saldo = float(X.loc[(y == 1).to_numpy(), "bill_amt_m1"].mean())
    print(f"Ventana: {len(X):,} filas ({(origen == 'entrenamiento_pset3').sum():,} del PSet 3 + "
          f"{(origen != 'entrenamiento_pset3').sum():,} de producción) | lote reservado {k_eval} ({len(y_ev)} clientes)")
    print("Pesos por lote:", {o: round(float(w[origen == o][0]), 2) for o in origen.unique() if o != "entrenamiento_pset3"})

    t0 = time.perf_counter()
    cvr = cv_challenger(champion, X, y, w, len(y_ev), saldo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        challenger = clone(champion).fit(X, y, modelo__sample_weight=w)
    seg = time.perf_counter() - t0

    e_ch = evaluar_politica(y_ev, champion.predict_proba(X_ev)[:, 1], saldo)
    e_cl = evaluar_politica(y_ev, challenger.predict_proba(X_ev)[:, 1], saldo)
    comp = chequeo_comportamiento(challenger, X_ev, max(cvr["corte_efectivo"], 0.05))
    delta = e_cl["ahorro"] - e_ch["ahorro"]
    sigma = cvr["cv_ahorro_lote_std"]
    tol_auc = champ_info.get("cv_roc_auc_std") or cvr["cv_roc_auc_std"]
    cumple_ahorro = delta > sigma
    cumple_auc = e_cl["auc"] >= e_ch["auc"] - tol_auc
    cumple = bool(cumple_ahorro and cumple_auc)
    texto = (f"Lote reservado {k_eval}: champion v{champ_info['version']} AUC {e_ch['auc']:.3f}, ahorro NT$ {e_ch['ahorro']:,.0f} | "
             f"challenger AUC {e_cl['auc']:.3f}, ahorro NT$ {e_cl['ahorro']:,.0f} | diferencia NT$ {delta:,.0f} vs "
             f"sigma_CV NT$ {sigma:,.0f} -> {'CUMPLE' if cumple else 'NO CUMPLE'} la regla de promoción | "
             f"pruebas de comportamiento: {comp['viol_atraso']} violaciones de atraso, {comp['viol_pago']} de pago")
    print(texto)

    with mlflow.start_run(run_name=f"reentrenamiento_lote_{lote}_{motivo}") as run:
        mlflow.set_tags({"fase": "reentrenamiento", "motivo": motivo, "lote_disparo": str(lote),
                         "lote_evaluacion": str(k_eval), "familia": champ_info.get("modelo") or "boosting",
                         "modelo": champ_info.get("modelo") or "champion", "git_commit": commit_git(),
                         "champion_previo_version": str(champ_info["version"]),
                         "champion_previo_run_id": champ_info["run_id"], "regla_promocion_cumple": str(cumple)})
        log_datos(meta)
        mlflow.log_params({"ventana": "historico_completo_con_pesos", "peso_max_lote_reciente": PESO_MAX,
                           "vida_media_lotes": VIDA_MEDIA, "n_entrenamiento": len(X),
                           "n_pset3": int((origen == "entrenamiento_pset3").sum()),
                           "lotes_produccion": ",".join(sorted({o.split("_")[1] for o in origen if o.startswith("lote_")},
                                                               key=int)) or "ninguno",
                           "umbral_decision": 0.05, "corte_efectivo_train_oof": round(cvr["corte_efectivo"], 4),
                           "saldo_medio_NT": round(saldo, 2), "t_exito": negocio.T_EXITO,
                           "c_contacto_NT": negocio.C_CONTACTO, "capacidad": negocio.CAPACIDAD,
                           "regla_promocion": "delta_ahorro > sigma_CV_ahorro y AUC_ch >= AUC_champ - cv_std",
                           "hiperparametros": "los del champion (clone, sin tuning)"})
        mlflow.log_metrics({"cv_roc_auc_mean": cvr["cv_roc_auc_mean"], "cv_roc_auc_std": cvr["cv_roc_auc_std"],
                            "cv_ahorro_lote_mean": cvr["cv_ahorro_lote_mean"], "cv_ahorro_lote_std": sigma,
                            "eval_auc_champion": e_ch["auc"], "eval_auc_challenger": e_cl["auc"],
                            "eval_ahorro_champion": e_ch["ahorro"], "eval_ahorro_challenger": e_cl["ahorro"],
                            "eval_delta_ahorro": delta, "train_seconds": seg,
                            "comportamiento_viol_atraso": comp["viol_atraso"],
                            "comportamiento_viol_pago": comp["viol_pago"]})
        log_modelo(challenger, X[feats], registrar=MODELO_REGISTRO)
        run_id = run.info.run_id
    version = int(c.search_model_versions(f"run_id='{run_id}'")[0].version)
    c.set_registered_model_alias(MODELO_REGISTRO, "challenger", str(version))
    c.set_model_version_tag(MODELO_REGISTRO, str(version), "motivo", motivo)
    c.set_model_version_tag(MODELO_REGISTRO, str(version), "regla_promocion_cumple", str(cumple))

    fila = {"lote_disparo": int(lote), "motivo": motivo, "lote_evaluacion": k_eval, "run_id": run_id,
            "version_challenger": version, "version_champion_previo": champ_info["version"],
            "auc_champion": e_ch["auc"], "auc_challenger": e_cl["auc"], "ahorro_champion": e_ch["ahorro"],
            "ahorro_challenger": e_cl["ahorro"], "delta_ahorro": delta, "sigma_cv_ahorro": sigma,
            "cumple_regla": int(cumple), "viol_comportamiento": comp["viol_atraso"] + comp["viol_pago"],
            "decision": "pendiente", "comentario": "", "decidido_por": "",
            "ts_entrenado": _ahora(), "ts_decision": ""}
    almacen.escribir("REENTRENAMIENTOS", pd.DataFrame([fila]), borrar_donde=f"version_challenger = {version}")
    _out(version=version, run_id=run_id, cumple_regla=cumple, resumen=texto, lote_evaluacion=k_eval,
         champion_version=champ_info["version"])
    return fila


def decidir(lote: int, version: int, aprobar: bool, comentario: str, por: str = "") -> dict:
    configurar()
    c = registro.cliente()
    r = almacen.leer(f"SELECT * FROM {almacen.ESQUEMA}.REENTRENAMIENTOS WHERE version_challenger = {int(version)}")
    fila = r.iloc[0].to_dict() if len(r) else {"lote_disparo": lote, "version_challenger": version}
    cumple = bool(int(fila.get("cumple_regla", 0)))
    if aprobar:
        res = registro.asignar_champion(version, c, motivo=f"aprobado: {comentario}")
        decision = "aprobado" if cumple else "aprobado_contra_regla"
        print(f"APROBADO: @champion -> v{version} (anterior v{res['anterior']} queda en @anterior)")
    else:
        decision = "rechazado"
        print(f"RECHAZADO: @champion se mantiene. Motivo: {comentario}")
    c.set_model_version_tag(MODELO_REGISTRO, str(version), "decision", decision)
    c.set_model_version_tag(MODELO_REGISTRO, str(version), "decision_comentario", comentario[:250])
    if fila.get("run_id"):
        c.set_tag(fila["run_id"], "decision", decision)
        c.set_tag(fila["run_id"], "decision_comentario", comentario[:250])
        c.set_tag(fila["run_id"], "decidido_por", por or "desconocido")
    fila.update({"decision": decision, "comentario": comentario, "decidido_por": por, "ts_decision": _ahora()})
    almacen.escribir("REENTRENAMIENTOS", pd.DataFrame([fila]), borrar_donde=f"version_challenger = {int(version)}")
    if aprobar and API_URL:
        registro._avisar_api()
    _out(decision=decision, version=version)
    return fila


def _bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "si", "sí", "yes"}


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=["entrenar", "decidir"])
    ap.add_argument("--lote", type=int, required=True)
    ap.add_argument("--motivo", default="manual")
    ap.add_argument("--version", type=int)
    ap.add_argument("--aprobar", default="false")
    ap.add_argument("--comentario", default="")
    ap.add_argument("--por", default="")
    a = ap.parse_args()
    if a.paso == "entrenar":
        entrenar(a.lote, a.motivo)
    else:
        decidir(a.lote, a.version, _bool(a.aprobar), a.comentario, a.por)


if __name__ == "__main__":
    _main()

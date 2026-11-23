"""
Ejecuta la simulación completa SIN Kestra, con los mismos pasos y en el mismo orden que los flujos
(útil para depurar y como respaldo si Kestra no está disponible). La aprobación humana se reemplaza
por la regla de promoción (aprueba si se cumple, rechaza si no) y queda marcada como automática.

  python scripts/simular_local.py              # prepara los lotes y corre del 1 al 10 + cierre
  python scripts/simular_local.py --sin-preparar --desde 4

Requiere la API levantada (API_URL) y MLflow (MLFLOW_TRACKING_URI).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoreo import alertas  # noqa: E402
from src import produccion, reentrenar, simulacion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-preparar", action="store_true")
    ap.add_argument("--desde", type=int, default=1)
    ap.add_argument("--hasta", type=int, default=simulacion.N_LOTES + 1)   # +1 = cierre (llegan los últimos targets)
    a = ap.parse_args()
    if not a.sin_preparar:
        produccion.preparar(rehacer=True)
    for lote in range(a.desde, a.hasta + 1):
        print(f"\n################ LOTE {lote} ({simulacion.periodo(lote)}) ################")
        produccion.puntuar(lote)
        produccion.volcar()
        produccion.unir(lote)
        r = alertas.monitorear(lote)
        if r["reentrenar"]:
            f = reentrenar.entrenar(lote, r["motivo"])
            aprobar = bool(f["cumple_regla"])
            reentrenar.decidir(lote, f["version_challenger"], aprobar,
                               "simulación local: decisión automática según la regla de promoción", por="simular_local")
            if aprobar:
                from src import registro

                registro._avisar_api()


if __name__ == "__main__":
    main()

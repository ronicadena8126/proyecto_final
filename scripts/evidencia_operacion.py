"""
Evidencia de rollback (sección 4) e interruptor de respaldo (sección 8.4), contra la API en Docker:

  python scripts/evidencia_operacion.py        -> docs/evidencia_operacion.json

1. GET /modelo con el champion actual.
2. Rollback: @champion vuelve a la versión de @anterior SOLO moviendo el alias en MLflow. La API lo
   toma sin reconstruir la imagen (se le pide recargar para no esperar los 30 s del sondeo).
3. Se restituye el champion original (misma operación en sentido contrario).
4. Interruptor: se activa el respaldo, una predicción sale con la regla de negocio, se desactiva.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from api.esquemas import Cliente  # noqa: E402
from src import registro  # noqa: E402

API = os.environ.get("API_URL", "http://localhost:8000")
TOK = {"X-Token": os.environ.get("ADMIN_TOKEN", "")}
EJEMPLO = Cliente.model_config["json_schema_extra"]["example"]


def paso(cli, titulo, ev):
    m = cli.get("/modelo").json()
    p = cli.post("/predecir", json=EJEMPLO).json()
    fila = {"paso": titulo, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "modelo": m,
            "prediccion": {k: p[k] for k in ("probabilidad", "decision", "version_modelo", "fuente", "respaldo",
                                             "motivo_respaldo", "id_prediccion")}}
    ev.append(fila)
    print(f"{titulo:<42} /modelo -> v{m['version']} ({m['fuente_activa']}) | /predecir -> "
          f"p={p['probabilidad']:.3f} {p['decision']} v{p['version_modelo']} fuente={p['fuente']}")


def main():
    ev = []
    with httpx.Client(base_url=API, timeout=120) as cli:
        paso(cli, "1. champion vigente", ev)
        original = registro.version_por_alias("champion").version
        r = registro.rollback()
        cli.post("/admin/recargar", headers=TOK).raise_for_status()
        paso(cli, f"2. rollback: alias -> v{r['champion']} (sin rebuild)", ev)
        registro.asignar_champion(original, motivo="restitución después de la prueba de rollback")
        cli.post("/admin/recargar", headers=TOK).raise_for_status()
        paso(cli, f"3. restitución: alias -> v{original}", ev)
        cli.post("/admin/respaldo", json={"activo": True, "motivo": "prueba del interruptor de respaldo"},
                 headers=TOK).raise_for_status()
        paso(cli, "4. interruptor de respaldo ENCENDIDO", ev)
        cli.post("/admin/respaldo", json={"activo": False}, headers=TOK).raise_for_status()
        paso(cli, "5. interruptor de respaldo APAGADO", ev)
    (RAIZ / "docs" / "evidencia_operacion.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    print("Guardado docs/evidencia_operacion.json")


if __name__ == "__main__":
    main()

"""Configuración del proyecto: rutas, constantes de dominio y estilo de los gráficos.

Todos los módulos de ``lib`` importan desde aquí, de modo que una sola definición
gobierna rutas, colores y semilla.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Rutas (calculadas desde la ubicación de este archivo, no desde el cwd)
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
FIG = RAIZ / "reports" / "figures"
TAB = RAIZ / "reports" / "tables"
PROCESADOS = RAIZ / "data" / "processed"

SEMILLA = 42

# --------------------------------------------------------------------------- #
# Paleta
# --------------------------------------------------------------------------- #
AZUL, NARANJA, ROJO = "#1f4e79", "#ed7d31", "#c00000"
VERDE, GRIS, CELESTE = "#548235", "#7f7f7f", "#2e75b6"
MOSTAZA, MORADO = "#bf9000", "#7030a0"

# --------------------------------------------------------------------------- #
# Constantes de dominio
# --------------------------------------------------------------------------- #
CAUSAS = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
NOMBRES_CAUSA = ["Aerolínea", "Clima", "Sistema aéreo", "Seguridad", "Aeronave previa"]
COLOR_CAUSA = dict(zip(CAUSAS, [AZUL, NARANJA, CELESTE, GRIS, VERDE]))

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# --------------------------------------------------------------------------- #
# Estilo y utilidades
# --------------------------------------------------------------------------- #
fmt_miles = FuncFormatter(lambda valor, _: f"{valor:,.0f}".replace(",", "."))


def aplicar_estilo() -> None:
    """Fija el estilo común de todas las figuras."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": .3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def guardar(fig, nombre: str) -> Path:
    """Guarda la figura en ``reports/figures`` y devuelve su ruta."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    ruta = FIG / nombre
    fig.savefig(ruta, bbox_inches="tight", dpi=150, facecolor="white")
    return ruta

"""Rutas, colores y constantes del proyecto."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# rutas
RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
FIG = RAIZ / "reports" / "figures"
TAB = RAIZ / "reports" / "tables"
PROCESADOS = RAIZ / "data" / "processed"

SEMILLA = 42

# colores
AZUL, NARANJA, ROJO = "#1f4e79", "#ed7d31", "#c00000"
VERDE, GRIS, CELESTE = "#548235", "#7f7f7f", "#2e75b6"
MOSTAZA, MORADO = "#bf9000", "#7030a0"

# constantes
CAUSAS = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
NOMBRES_CAUSA = ["Aerolínea", "Clima", "Sistema aéreo", "Seguridad", "Aeronave previa"]
COLOR_CAUSA = dict(zip(CAUSAS, [AZUL, NARANJA, CELESTE, GRIS, VERDE]))

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# estilo
fmt_miles = FuncFormatter(lambda valor, _: f"{valor:,.0f}".replace(",", "."))


def aplicar_estilo() -> None:
    """Estilo de los gráficos."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": .3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def guardar(fig, nombre: str) -> Path:
    """Guarda la figura."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    ruta = FIG / nombre
    fig.savefig(ruta, bbox_inches="tight", dpi=150, facecolor="white")
    return ruta

"""Pruebas de calidad: registro y utilidades."""

from __future__ import annotations

import pandas as pd


def cnt(mascara) -> int:
    """Cuenta True sin contar NA."""
    return int(pd.Series(mascara).fillna(False).sum())


def minutos_hhmm(serie: pd.Series) -> pd.Series:
    """Pasa HHMM a minutos."""
    valores = serie.astype("Float64")
    return (valores // 100) * 60 + valores % 100


class RegistroCalidad:
    """Guarda las pruebas de calidad en una tabla."""

    COLUMNAS = ["codigo", "dimension", "prueba", "afectados", "tasa_%",
                "umbral_%", "estado", "severidad", "accion"]

    def __init__(self) -> None:
        self.filas: list[dict] = []

    def check(self, cod: str, dim: str, prueba: str, malos: int, total: int,
              umbral: float = 0.0, sev: str = "Alta", accion: str = "—") -> "RegistroCalidad":
        """Añade una prueba."""
        tasa = 100 * malos / total if total else 0.0
        self.filas.append({
            "codigo": cod, "dimension": dim, "prueba": prueba,
            "afectados": int(malos), "tasa_%": round(tasa, 4), "umbral_%": umbral,
            "estado": "OK" if tasa <= umbral else "REVISAR",
            "severidad": sev, "accion": accion,
        })
        return self

    @property
    def tabla(self) -> pd.DataFrame:
        """Tabla con todas las pruebas."""
        return pd.DataFrame(self.filas, columns=self.COLUMNAS)

    def resumen(self) -> pd.DataFrame:
        """Resumen por dimensión."""
        tabla = self.tabla
        resumen = tabla.groupby("dimension").agg(
            pruebas=("codigo", "size"),
            ok=("estado", lambda s: int((s == "OK").sum())),
        )
        resumen["cumple_%"] = (resumen["ok"] / resumen["pruebas"] * 100).round(0)
        return resumen

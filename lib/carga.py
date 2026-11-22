"""Carga de las fuentes del proyecto.

La tabla de hechos tiene 11,7 M de filas y 40 columnas; leerla completa consume
varios GB. Aquí se proyectan sólo las columnas necesarias y se construye una
muestra estratificada por mes para los gráficos de nivel de fila.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from config import RAW, SEMILLA

COLS = [
    "flight_date", "month", "day_of_week", "airline", "tail_number", "flight_number",
    "origin", "origin_state", "dest", "dest_state", "sched_dep_time", "dep_time",
    "dep_delay", "dep_del15", "taxi_out", "sched_arr_time", "arr_time", "taxi_in",
    "arr_delay", "arr_del15", "cancelled", "cancellation_code", "cancellation_reason",
    "diverted", "sched_elapsed_time", "actual_elapsed_time", "air_time", "distance",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
]

FRACCION_MUESTRA = 1 / 15


def inventario_fuentes(carpeta=RAW) -> pd.DataFrame:
    """Archivo, filas, columnas y tamaño de cada fuente de ``data/raw``."""
    filas = []
    for archivo in sorted(carpeta.iterdir()):
        if archivo.suffix == ".parquet":
            meta = pq.ParquetFile(archivo).metadata
            n, c = meta.num_rows, meta.num_columns
        else:
            n = sum(1 for _ in open(archivo, "rb")) - 1
            c = pd.read_csv(archivo, nrows=0).shape[1]
        filas.append({
            "archivo": archivo.name,
            "filas": n,
            "columnas": c,
            "MB": round(archivo.stat().st_size / 1024**2, 1),
        })
    return pd.DataFrame(filas)


def cargar_vuelos(fraccion: float = FRACCION_MUESTRA, semilla: int = SEMILLA):
    """Devuelve ``(df, muestra)``: tabla completa proyectada y muestra por mes."""
    df = pq.read_table(RAW / "flights.parquet", columns=COLS).to_pandas()
    df["fecha"] = pd.to_datetime(df["flight_date"])
    df["hora"] = (df["sched_dep_time"] // 100).clip(0, 23).astype("int8")

    muestra = (
        df.groupby("month", group_keys=False, observed=True)
          .sample(frac=fraccion, random_state=semilla)
          .reset_index(drop=True)
    )
    return df, muestra

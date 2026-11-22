"""Gráficos del análisis exploratorio.

Cada función recibe los datos ya agregados en el cuadernillo, construye la
figura, la guarda en ``reports/figures`` y la devuelve para mostrarla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    AZUL, NARANJA, ROJO, VERDE, GRIS, CELESTE, MORADO, MOSTAZA,
    CAUSAS, NOMBRES_CAUSA, COLOR_CAUSA, DIAS, fmt_miles, guardar,
)


def distribuciones(df: pd.DataFrame):
    """Histogramas de las variables continuas con media y mediana."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    configuracion = [
        ("dep_delay", (-60, 180), AZUL), ("arr_delay", (-60, 180), NARANJA),
        ("distance", None, CELESTE), ("air_time", (0, 400), VERDE),
        ("taxi_out", (0, 60), MORADO), ("sched_elapsed_time", (0, 450), GRIS),
    ]
    for eje, (columna, rango, color) in zip(ax.ravel(), configuracion):
        datos = df[columna].astype("float64").dropna()
        datos = datos[datos.between(*rango)] if rango else datos
        eje.hist(datos, bins=60, color=color, alpha=.85, edgecolor="white", linewidth=.3)
        eje.set_title(columna)
        eje.set_ylabel("vuelos")
        eje.yaxis.set_major_formatter(fmt_miles)
        if columna in ("dep_delay", "arr_delay"):
            eje.axvline(df[columna].median(), color=ROJO, label=f"mediana {df[columna].median():.0f}")
            eje.axvline(df[columna].mean(), color=MOSTAZA, ls="--", label=f"media {df[columna].mean():.1f}")
            eje.legend(fontsize=8)
    fig.suptitle("Distribuciones de las variables numéricas", y=1.0)
    guardar(fig, "01_distribuciones.png")
    return fig


def categoricas(df: pd.DataFrame):
    """Volumen por aerolínea, aeropuerto, mes y día de la semana."""
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 8.5))

    conteo = df["airline"].value_counts().sort_values()
    ax[0, 0].barh(conteo.index, conteo.values, color=AZUL)
    ax[0, 0].set_title("Vuelos por aerolínea")
    ax[0, 0].xaxis.set_major_formatter(fmt_miles)
    ax[0, 0].set_xlabel("vuelos")

    conteo = df["origin"].value_counts().head(12).sort_values()
    ax[0, 1].barh(conteo.index, conteo.values, color=NARANJA)
    ax[0, 1].set_title("Top 12 aeropuertos")
    ax[0, 1].xaxis.set_major_formatter(fmt_miles)
    ax[0, 1].set_xlabel("salidas")

    por_mes = df.groupby("month", observed=True).size()
    ax[1, 0].bar(por_mes.index, por_mes.values, color=CELESTE)
    ax[1, 0].set_title("Vuelos por mes")
    ax[1, 0].set_xticks(range(1, 13))
    ax[1, 0].yaxis.set_major_formatter(fmt_miles)

    por_dia = df.groupby("day_of_week", observed=True).size()
    ax[1, 1].bar(DIAS, por_dia.values, color=VERDE)
    ax[1, 1].set_title("Vuelos por día")
    ax[1, 1].yaxis.set_major_formatter(fmt_miles)

    guardar(fig, "02_categoricas.png")
    return fig


def aerolineas(aer: pd.DataFrame, media: float, tasa_global: float):
    """Retraso medio y tasa de llegada tardía por aerolínea."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    orden = aer.sort_values("retraso")
    ax[0].barh(orden.index, orden["retraso"],
               color=[ROJO if v > media else VERDE for v in orden["retraso"]])
    ax[0].axvline(media, color=AZUL, ls="--", label=f"media {media:.1f} min")
    ax[0].set_title("Retraso medio de salida")
    ax[0].set_xlabel("min")
    ax[0].legend(fontsize=8)

    orden = aer.sort_values("tarde")
    ax[1].barh(orden.index, orden["tarde"], color=NARANJA)
    ax[1].axvline(tasa_global, color=AZUL, ls="--")
    ax[1].set_title("Llegadas con ≥ 15 min de retraso")
    ax[1].set_xlabel("%")

    guardar(fig, "03_aerolineas.png")
    return fig


def aeropuertos(grandes: pd.DataFrame, extremos: pd.DataFrame, media: float):
    """Mejores y peores aeropuertos y relación volumen-tasa."""
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.5))

    ax[0].barh(extremos.index, extremos["retraso"],
               color=[ROJO if v > media else VERDE for v in extremos["retraso"]])
    ax[0].axvline(media, color=AZUL, ls="--")
    ax[0].set_title("Mejores y peores aeropuertos (≥ 20k salidas)")
    ax[0].set_xlabel("retraso medio (min)")

    dispersion = ax[1].scatter(grandes["vuelos"], grandes["tarde"],
                               s=grandes["cancela"] * 20 + 15,
                               c=grandes["retraso"], cmap="YlOrRd",
                               edgecolor="white", linewidth=.5)
    ax[1].set_xscale("log")
    ax[1].set_title("Volumen frente a tasa de retraso")
    ax[1].set_xlabel("salidas (log)")
    ax[1].set_ylabel("% llegadas tardías")
    for codigo in ["DFW", "ORD", "LIH", "DEN"]:
        ax[1].annotate(codigo, (grandes.loc[codigo, "vuelos"], grandes.loc[codigo, "tarde"]),
                       fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(dispersion, ax=ax[1], label="retraso medio")

    guardar(fig, "04_aeropuertos.png")
    return fig


def horario(por_hora: pd.DataFrame, por_dia: pd.DataFrame):
    """Perfil horario y semanal del retraso."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    ax[0, 0].plot(por_hora.index, por_hora["media"], "o-", color=AZUL, label="media")
    ax[0, 0].plot(por_hora.index, por_hora["mediana"], "s-", color=NARANJA, label="mediana")
    ax[0, 0].axhline(0, color=GRIS, lw=1)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("Retraso por hora programada")
    ax[0, 0].set_xlabel("hora local")
    ax[0, 0].set_ylabel("min")

    ax[0, 1].bar(por_hora.index, por_hora["vuelos"], color="#d9e2f3", width=.8)
    gemelo = ax[0, 1].twinx()
    gemelo.plot(por_hora.index, por_hora["tarde"], "o-", color=ROJO)
    gemelo.grid(False)
    gemelo.set_ylabel("% con retraso", color=ROJO)
    ax[0, 1].set_xlabel("hora local")
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title("Tasa de retraso por hora")

    ax[1, 0].bar(DIAS, por_dia["media"],
                 color=[NARANJA if i > 4 else AZUL for i in range(7)])
    ax[1, 0].set_title("Retraso medio por día")
    ax[1, 0].set_ylabel("min")

    ax[1, 1].bar(DIAS, por_dia["tarde"], color=VERDE)
    ax[1, 1].set_title("Tasa de retraso por día")
    ax[1, 1].set_ylabel("%")

    guardar(fig, "05_horario.png")
    return fig


def temporal(dia: pd.DataFrame, mes_hora: pd.DataFrame, conteo_mes_hora: pd.DataFrame,
             media: float):
    """Serie diaria, cancelaciones y mapa de calor mes × hora."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    ax[0, 0].plot(dia.index, dia["media"], color="#c9d6e6", lw=.8)
    ax[0, 0].plot(dia.index, dia["media"].rolling(7, center=True, min_periods=4).mean(),
                  color=AZUL, lw=2)
    ax[0, 0].axhline(media, color=ROJO, ls=":")
    ax[0, 0].set_title("Retraso medio diario (+ media 7 días)")

    ax[1, 0].fill_between(dia.index, 0, dia["cancela"] * 100, color=NARANJA, alpha=.4)
    ax[1, 0].plot(dia.index, (dia["cancela"] * 100).rolling(7, center=True, min_periods=4).mean(),
                  color=ROJO)
    ax[1, 0].set_title("Cancelaciones diarias (%)")

    mapa = ax[0, 1].imshow(mes_hora.where(conteo_mes_hora >= 100), aspect="auto",
                           cmap="YlOrRd", origin="lower")
    ax[0, 1].set_title("Retraso medio mes × hora")
    ax[0, 1].set_yticks(range(12), range(1, 13))
    ax[0, 1].grid(False)
    plt.colorbar(mapa, ax=ax[0, 1], label="min")

    ax[1, 1].axis("off")
    ax[1, 1].text(
        0, .5,
        f"día más caótico: {dia['media'].idxmax():%Y-%m-%d} ({dia['media'].max():.1f} min)\n"
        f"día más puntual: {dia['media'].idxmin():%Y-%m-%d} ({dia['media'].min():.1f} min)\n"
        f"días con media > 30 min: {(dia['media'] > 30).sum()}\n"
        f"días con cancelación > 5 %: {(dia['cancela'] > .05).sum()}",
        fontsize=11, va="center",
    )

    guardar(fig, "06_temporal.png")
    return fig


def rutas(muestra: pd.DataFrame, df: pd.DataFrame):
    """Distancia frente a duración y retraso por tramo de distancia."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    hexbin = ax[0].hexbin(muestra["distance"], muestra["air_time"],
                          gridsize=50, cmap="Blues", mincnt=1)
    r = muestra[["distance", "air_time"]].corr().iloc[0, 1]
    ax[0].set_title(f"Distancia vs duración (r = {r:.3f})")
    ax[0].set_xlabel("millas")
    ax[0].set_ylabel("min en el aire")
    plt.colorbar(hexbin, ax=ax[0])

    cortes = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 6000]
    tramos = pd.cut(df["distance"], cortes,
                    labels=["<250", "250-500", "500-750", "750-1k",
                            "1k-1.5k", "1.5k-2k", "2k-3k", ">3k"])
    retraso = df.groupby(tramos, observed=True).dep_delay.mean()
    ax[1].bar(retraso.index.astype(str), retraso.values, color=AZUL)
    ax[1].set_title("Retraso medio por tramo de distancia")
    ax[1].set_ylabel("min")
    ax[1].tick_params(axis="x", rotation=30)

    guardar(fig, "07_rutas.png")
    return fig


def causas(parte: pd.Series, reparto: pd.DataFrame):
    """Peso de cada causa y reparto por aerolínea."""
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5))

    orden = parte.sort_values()
    ax[0].barh([NOMBRES_CAUSA[CAUSAS.index(c)] for c in orden.index], orden.values,
               color=[COLOR_CAUSA[c] for c in orden.index])
    ax[0].set_title("Minutos de retraso atribuidos (%)")
    ax[0].set_xlabel("%")

    izquierda = np.zeros(len(reparto))
    for causa, nombre in zip(CAUSAS, NOMBRES_CAUSA):
        ax[1].barh(reparto.index, reparto[causa], left=izquierda,
                   color=COLOR_CAUSA[causa], label=nombre)
        izquierda += reparto[causa].values
    ax[1].set_title("Reparto de causas por aerolínea")
    ax[1].set_xlim(0, 100)
    ax[1].legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))

    guardar(fig, "08_causas.png")
    return fig


def correlaciones(muestra: pd.DataFrame, variables: list[str]):
    """Matriz de Spearman y dispersión salida-llegada."""
    spearman = muestra[variables].astype("float64").corr(method="spearman")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))

    imagen = ax[0].imshow(spearman, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(spearman)), spearman.columns, rotation=45, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(spearman)), spearman.columns, fontsize=8)
    ax[0].set_title("Spearman")
    for i in range(1, len(spearman)):
        for j in range(i):
            ax[0].text(j, i, f"{spearman.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(imagen, ax=ax[0], shrink=.85)

    pares = muestra[["dep_delay", "arr_delay"]].dropna().astype("float64")
    pares = pares[pares.dep_delay.between(-30, 120) & pares.arr_delay.between(-30, 120)]
    ax[1].hexbin(pares.dep_delay, pares.arr_delay, gridsize=45, cmap="Blues", mincnt=1)
    ax[1].plot([-30, 120], [-30, 120], color=ROJO, ls="--", lw=1, label="llegada = salida")
    ax[1].set_xlabel("retraso de salida (min)")
    ax[1].set_ylabel("retraso de llegada (min)")
    pearson = muestra[["dep_delay", "arr_delay"]].corr().iloc[0, 1]
    ax[1].set_title(f"Salida vs llegada (Pearson {pearson:.3f})")
    ax[1].legend(fontsize=8)

    guardar(fig, "09_correlaciones.png")
    return fig


def atipicos(df: pd.DataFrame, umbrales: pd.DataFrame):
    """Cajas por mes y conteo de vuelos que superan cada umbral."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))

    datos = [df.loc[df.month == mes, "dep_delay"].dropna().astype("float64") for mes in range(1, 13)]
    ax[0].boxplot(datos, tick_labels=[f"{m:02d}" for m in range(1, 13)],
                  showfliers=False, patch_artist=True)
    ax[0].set_title("Retraso por mes (sin extremos)")
    ax[0].set_ylabel("min")
    ax[0].set_ylim(-40, 120)

    ax[1].bar(umbrales["umbral (min)"].astype(str), umbrales["vuelos"], color=NARANJA)
    ax[1].set_yscale("log")
    ax[1].set_title("Vuelos que superan cada umbral (log)")
    ax[1].set_xlabel("min")

    guardar(fig, "10_atipicos.png")
    return fig


def tablero(por_dimension: pd.DataFrame, registro: pd.DataFrame):
    """Cumplimiento por dimensión y registros afectados por prueba."""
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8))

    ax[0].barh(por_dimension.index, por_dimension["cumple_%"],
               color=[VERDE if v == 100 else MOSTAZA for v in por_dimension["cumple_%"]])
    ax[0].axvline(100, color=GRIS, ls="--")
    ax[0].set_xlim(0, 110)
    ax[0].set_title("Cumplimiento por dimensión (%)")
    ax[0].set_xlabel("%")

    afectadas = registro[registro.afectados > 0]
    ax[1].scatter(afectadas["tasa_%"], afectadas["codigo"],
                  c=[ROJO if e == "REVISAR" else AZUL for e in afectadas.estado], s=60)
    ax[1].set_xscale("log")
    ax[1].set_title("Registros afectados por prueba (%)")
    ax[1].invert_yaxis()

    guardar(fig, "11_tablero.png")
    return fig

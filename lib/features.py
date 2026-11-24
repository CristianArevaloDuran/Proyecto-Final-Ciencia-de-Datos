"""Preparar los datos: derivadas y codificación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TRAMOS = ["00-05", "06-08", "09-11", "12-14", "15-17", "18-20", "21-23"]
CORTES = [-1, 5, 8, 11, 14, 17, 20, 23]
CATEGORICAS = ["airline", "origin", "dest", "tramo_hora"]
NUMERICAS = ["hora", "month", "day_of_week", "es_hub", "salidas_hora",
             "distance", "retraso_previo", "sin_previo"]
CICLICAS = [("hora", 24), ("month", 12), ("day_of_week", 7)]


def preparar_base(df: pd.DataFrame, top_hubs: int = 20) -> pd.DataFrame:
    """Filtra vuelos con etiqueta y crea las derivadas."""
    d = df[df["arr_del15"].notna()].copy()

    d["tramo_hora"] = pd.cut(d["hora"], CORTES, labels=TRAMOS)
    hubs = d["origin"].value_counts().head(top_hubs).index
    d["es_hub"] = d["origin"].isin(hubs).astype("int8")

    d["salidas_hora"] = (d.groupby(["origin", "flight_date", "hora"], observed=True)
                          .flight_date.transform("size"))
    d["salidas_hora"] = np.log1p(d["salidas_hora"])

    d = d.sort_values(["tail_number", "flight_date", "hora"])
    d["retraso_previo"] = d.groupby("tail_number")["arr_delay"].shift()
    d["sin_previo"] = d["retraso_previo"].isna().astype("int8")
    d["retraso_previo"] = d["retraso_previo"].fillna(0)
    return d.reset_index(drop=True)


def dividir_temporal(d: pd.DataFrame, corte: str = "2026-01-01"):
    """Corta por fecha."""
    return d[d["fecha"] < corte], d[d["fecha"] >= corte]


def codificar_ordinal(d: pd.DataFrame) -> pd.DataFrame:
    """Códigos enteros para comparar modelos rápido."""
    salida = d[["arr_del15", "fecha"] + NUMERICAS + CATEGORICAS].copy()
    for col in CATEGORICAS:
        salida[col] = salida[col].astype("category").cat.codes
    return salida


def target_encoding(train: pd.DataFrame, test: pd.DataFrame, col: str,
                    target: str = "arr_del15", suavizado: float = 20,
                    folds: int = 5, semilla: int = 42):
    """Target encoding con suavizado (sin fuga)."""
    media = train[target].mean()
    train_enc = pd.Series(index=train.index, dtype="float64")

    for tr, va in KFold(folds, shuffle=True, random_state=semilla).split(train):
        agg = train.iloc[tr].groupby(col, observed=True)[target].agg(["mean", "size"])
        mapa = (agg["mean"] * agg["size"] + media * suavizado) / (agg["size"] + suavizado)
        valores = (train.iloc[va][col].map(mapa).fillna(media)
                   .to_numpy(dtype="float64"))
        train_enc.iloc[va] = valores

    agg = train.groupby(col, observed=True)[target].agg(["mean", "size"])
    mapa = (agg["mean"] * agg["size"] + media * suavizado) / (agg["size"] + suavizado)
    return train_enc.fillna(media), test[col].map(mapa).fillna(media)


def construir_matrices(train: pd.DataFrame, test: pd.DataFrame,
                       target: str = "arr_del15"):
    """Devuelve X e y ya codificados."""
    # one-hot
    juntos = pd.concat([train[["airline", "tramo_hora", "es_hub"]],
                        test[["airline", "tramo_hora", "es_hub"]]])
    dummies = pd.get_dummies(juntos, columns=["airline", "tramo_hora"],
                             drop_first=True, dtype="int8")
    oh_train, oh_test = dummies.iloc[:len(train)], dummies.iloc[len(train):]

    # seno y coseno
    def ciclo(d):
        salida = pd.DataFrame(index=d.index)
        for col, periodo in CICLICAS:
            salida[f"{col}_sin"] = np.sin(2 * np.pi * d[col] / periodo)
            salida[f"{col}_cos"] = np.cos(2 * np.pi * d[col] / periodo)
        return salida

    # target encoding
    te_tr = pd.DataFrame(index=train.index)
    te_te = pd.DataFrame(index=test.index)
    for col in ["origin", "dest"]:
        te_tr[f"{col}_te"], te_te[f"{col}_te"] = target_encoding(train, test, col, target)

    base_tr = train[["hora", "month", "day_of_week", "salidas_hora",
                     "distance", "retraso_previo", "sin_previo"]]
    base_te = test[["hora", "month", "day_of_week", "salidas_hora",
                    "distance", "retraso_previo", "sin_previo"]]

    X_train = pd.concat([base_tr, oh_train, ciclo(train), te_tr], axis=1).astype("float32")
    X_test = pd.concat([base_te, oh_test, ciclo(test), te_te], axis=1).astype("float32")
    return X_train, X_test, train[target], test[target]

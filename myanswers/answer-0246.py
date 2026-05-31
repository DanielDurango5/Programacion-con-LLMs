import pandas as pd


def calcular_metricas_campana(df):

    # Copia defensiva
    df_calc = df.copy()

    # Calcular CTR
    df_calc["ctr"] = (
        df_calc["clicks"] / df_calc["impresiones"]
    )

    # Calcular CPC
    df_calc["cpc"] = (
        df_calc["costo"] / df_calc["clicks"]
    )

    # Agrupar y calcular promedios
    resultado = (
        df_calc.groupby("campaña")[["ctr", "cpc"]]
        .mean()
        .reset_index()
        .rename(columns={
            "ctr": "ctr_promedio",
            "cpc": "cpc_promedio"
        })
    )

    return resultado


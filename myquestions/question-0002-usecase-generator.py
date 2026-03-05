import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_crear_lag_features():

    n_rows = random.randint(10,20)
    values = np.random.randn(n_rows)

    df = pd.DataFrame({"serie": values})

    n_lags = random.randint(1,3)

    input_data = {
        "df": df.copy(),
        "columna": "serie",
        "n_lags": n_lags
    }

    df_expected = df.copy()

    for i in range(1, n_lags+1):
        df_expected[f"serie_lag_{i}"] = df_expected["serie"].shift(i)

    df_expected = df_expected.dropna()

    output_data = df_expected

    return input_data, output_data
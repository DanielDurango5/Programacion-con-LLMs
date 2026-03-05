import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_seleccionar_por_correlacion():

    n_rows = random.randint(10,20)
    n_features = random.randint(4,6)

    data = np.random.randn(n_rows, n_features)

    cols = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(data, columns=cols)

    target_col = "target"
    df[target_col] = np.random.randn(n_rows)

    k = random.randint(1,3)

    input_data = {
        "df": df.copy(),
        "target_col": target_col,
        "k": k
    }

    corr = df.corr()[target_col].abs().drop(target_col)

    top_features = corr.sort_values(ascending=False).head(k).index

    output_data = df[top_features]

    return input_data, output_data
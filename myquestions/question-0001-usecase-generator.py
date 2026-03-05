import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_eliminar_outliers_iqr():

    n_rows = random.randint(10,20)
    n_cols = random.randint(2,4)

    data = np.random.randn(n_rows, n_cols)

    # introducir algunos outliers
    for _ in range(random.randint(1,3)):
        r = random.randint(0,n_rows-1)
        c = random.randint(0,n_cols-1)
        data[r,c] = data[r,c] * 10

    cols = [f"feature_{i}" for i in range(n_cols)]
    df = pd.DataFrame(data, columns=cols)

    input_data = {"df": df.copy()}

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)

    output_data = df[mask]

    return input_data, output_data
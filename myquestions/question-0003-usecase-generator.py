import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import QuantileTransformer

def generar_caso_de_uso_transformar_cuantiles():

    n_rows = random.randint(10,20)
    n_cols = random.randint(2,4)

    data = np.random.randn(n_rows, n_cols)

    cols = [f"feature_{i}" for i in range(n_cols)]

    df = pd.DataFrame(data, columns=cols)

    input_data = {"df": df.copy()}

    transformer = QuantileTransformer(output_distribution="uniform")

    X_transformed = transformer.fit_transform(df)

    output_data = X_transformed

    return input_data, output_data
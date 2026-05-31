import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def reducir_dimensionalidad_tsne(df, columnas, random_state):

    # 1. Seleccionar columnas
    X = df[columnas].values

    # 2. Estandarizar datos
    X_scaled = StandardScaler().fit_transform(X)

    # 3. Definir perplexity igual al generador
    n_samples = X_scaled.shape[0]

    perplexity = min(5, n_samples - 1)

    # 4. Aplicar TSNE
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity
    )

    resultado = tsne.fit_transform(X_scaled)

    # 5. Retornar array transformado
    return resultado


import numpy as np
from sklearn.linear_model import Ridge


def weighted_regression_score(X, y, weights):

    # Entrenar modelo Ridge
    model = Ridge()

    model.fit(X, y)

    # Generar predicciones
    y_pred = model.predict(X)

    # Calcular Weighted Mean Squared Error
    wmse = (
        np.sum(weights * (y - y_pred) ** 2)
        / np.sum(weights)
    )

    # Retornar float
    return float(wmse)


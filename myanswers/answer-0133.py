import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture


def seleccionar_gmm_por_bic(
    df,
    features=None,
    k_values=(1, 2, 3, 4, 5),
    covariance_type='full',
    random_state=42
):

    # ---------------------------------------------------------
    # 1. Seleccionar columnas
    # ---------------------------------------------------------
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[features].to_numpy()

    # ---------------------------------------------------------
    # 2. Preprocesamiento
    # ---------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # ---------------------------------------------------------
    # 3. Ajustar GMMs y calcular BIC
    # ---------------------------------------------------------
    bic_rows = []
    gmms = {}

    for k in k_values:

        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type=covariance_type,
            random_state=int(random_state),
            n_init=2
        )

        gmm.fit(X_scaled)

        bic = gmm.bic(X_scaled)

        gmms[int(k)] = gmm

        bic_rows.append(
            (int(k), float(bic))
        )

    bic_df = pd.DataFrame(
        bic_rows,
        columns=["k", "bic"]
    ).sort_values("k").reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Seleccionar mejor modelo
    # ---------------------------------------------------------
    best_k = int(
        bic_df.sort_values("bic", ascending=True)
        .iloc[0]["k"]
    )

    best_gmm = gmms[best_k]

    labels = best_gmm.predict(X_scaled).astype(int)

    # ---------------------------------------------------------
    # 5. Pipeline final
    # ---------------------------------------------------------
    final_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=covariance_type,
        random_state=int(random_state),
        n_init=2
    )

    pipe = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler),
        ("gmm", final_gmm)
    ])

    pipe.fit(X)

    # ---------------------------------------------------------
    # 6. Return
    # ---------------------------------------------------------
    return labels, best_k, bic_df, pipe


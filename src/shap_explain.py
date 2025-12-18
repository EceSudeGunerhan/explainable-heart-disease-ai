import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing import (
    load_data,
    remove_duplicates,
    clip_outliers,
    split_features_target,
    create_preprocessor,
    split_train_test
)


def train_logistic_model(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]
    )
    model.fit(X_train, y_train)
    return model


def compute_shap(model, X_train, X_sample):
    """
    SHAP değerlerini hesaplar.
    """
    # Pipeline içinden dönüştürülmüş veriyi al
    X_train_transformed = model.named_steps["preprocessor"].transform(X_train)
    X_sample_transformed = model.named_steps["preprocessor"].transform(X_sample)

    explainer = shap.Explainer(
        model.named_steps["model"],
        X_train_transformed
    )

    shap_values = explainer(X_sample_transformed)
    return shap_values


if __name__ == "__main__":
    DATA_PATH = "../data/heart.csv"

    # Veri ve preprocessing
    df = load_data(DATA_PATH)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    X, y = split_features_target(df)
    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Model
    model = train_logistic_model(X_train, y_train, preprocessor)

    # Testten küçük bir örnek (ilk 10 kişi)
    X_sample = X_test.iloc[:10]

    shap_values = compute_shap(model, X_train, X_sample)

    # Özet grafik
    shap.plots.bar(shap_values)

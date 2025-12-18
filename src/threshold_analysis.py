import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import (
    load_data,
    remove_duplicates,
    clip_outliers,
    split_features_target,
    create_preprocessor,
    split_train_test
)


def threshold_analysis(model, X_test, y_test, thresholds):
    y_proba = model.predict_proba(X_test)[:, 1]

    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        results.append({
            "threshold": t,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    DATA_PATH = "../data/heart.csv"

    # Veri ve preprocessing
    df = load_data(DATA_PATH)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    X, y = split_features_target(df)
    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Logistic Regression modeli
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

    thresholds = np.arange(0.3, 0.61, 0.05)
    results = threshold_analysis(model, X_test, y_test, thresholds)

    print(results)



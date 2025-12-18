import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from preprocessing import (
    load_data,
    remove_duplicates,
    clip_outliers,
    split_features_target,
    create_preprocessor,
    split_train_test
)


def train_logistic_regression(X_train, y_train, preprocessor):
    """
    Logistic Regression modeli için pipeline oluşturur ve eğitir.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Model performans metriklerini hesaplar.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics


if __name__ == "__main__":
    DATA_PATH = "../data/heart.csv"

    # Preprocessing
    df = load_data(DATA_PATH)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    X, y = split_features_target(df)
    preprocessor = create_preprocessor()

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Model eğitimi
    model = train_logistic_regression(X_train, y_train, preprocessor)

    # Değerlendirme
    metrics = evaluate_model(model, X_test, y_test)

    print("Logistic Regression Sonuçları")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k}: \n{v}\n")

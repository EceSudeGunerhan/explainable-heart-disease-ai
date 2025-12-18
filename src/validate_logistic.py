import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score

from preprocessing import (
    load_data,
    remove_duplicates,
    clip_outliers,
    split_features_target,
    create_preprocessor
)


def cross_validate_logistic(X, y, preprocessor, cv=5):
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

    recall_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="recall"
    )

    roc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc"
    )

    return recall_scores, roc_scores


if __name__ == "__main__":
    DATA_PATH = "../data/heart.csv"

    df = load_data(DATA_PATH)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    X, y = split_features_target(df)
    preprocessor = create_preprocessor()

    recall_scores, roc_scores = cross_validate_logistic(X, y, preprocessor)

    print("Cross-Validation Sonuçları (Logistic Regression)")
    print("-" * 45)
    print(f"Recall: mean={recall_scores.mean():.3f}, std={recall_scores.std():.3f}")
    print(f"ROC-AUC: mean={roc_scores.mean():.3f}, std={roc_scores.std():.3f}")



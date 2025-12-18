import joblib
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

if __name__ == "__main__":
    DATA_PATH = "../data/heart.csv"
    MODEL_PATH = "../model.pkl"

    df = load_data(DATA_PATH)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    X, y = split_features_target(df)
    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

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

    joblib.dump(model, MODEL_PATH)
    print("Model kaydedildi -> model.pkl")

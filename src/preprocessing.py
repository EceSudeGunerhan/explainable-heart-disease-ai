import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def load_data(path: str) -> pd.DataFrame:
    """
    CSV veri setini yükler.
    """
    df = pd.read_csv(path)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mükerrer (duplicate) satırları siler.
    """
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]

    print(f"Duplicate temizleme: {before - after} satır silindi.")
    return df

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aykırı değerleri IQR yöntemine göre baskılar (clipping).
    Özellikle chol, trestbps ve oldpeak için uygulanır.
    """
    df = df.copy()

    cols = ["chol", "trestbps", "oldpeak"]

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower, upper=upper)

        print(
            f"{col} için clipping uygulandı "
            f"(lower={lower:.2f}, upper={upper:.2f})"
        )

    return df

def split_features_target(df: pd.DataFrame):
    """
    Özellikler (X) ve hedef değişkeni (y) ayırır.
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def create_preprocessor():
    """
    Sayısal ve kategorik değişkenler için preprocessing pipeline oluşturur.
    """
    numeric_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ]

    categorical_features = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test setine böler (stratify korunur).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test




if __name__ == "__main__":
    DATA_PATH = "../data/heart.csv"

    df = load_data(DATA_PATH)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    X, y = split_features_target(df)
    preprocessor = create_preprocessor()

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("Preprocessing tamamlandı.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


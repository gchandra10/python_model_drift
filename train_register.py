# train_register.py

import hashlib
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TRACKING_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "housing_training"

MODEL_NAME = "housing_price_model"
REGISTERED_MODEL = "housing_price_model"

TRAIN_DATA_URL = "https://raw.githubusercontent.com/gchandra10/filestorage/refs/heads/main/Housing.csv"

SEED = 42

FEATURES_V1 = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]

# ---------------- MLflow Helpers ----------------
def setup_mlflow(uri, exp):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)

def log_params():
    mlflow.log_params({
        "model": "LinearRegression",
        "features": json.dumps(FEATURES_V1),
        "features_hash": hashlib.md5(",".join(FEATURES_V1).encode()).hexdigest(),
        "n_features": len(FEATURES_V1),
        "scaler": "StandardScaler",
        "imputer": "SimpleImputer(median)",
        "dataset_url": TRAIN_DATA_URL,
    })

class Metrics:
    def __init__(self, y, y_pred):
        self.mae = mean_absolute_error(y, y_pred)
        self.mse = mean_squared_error(y, y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(y, y_pred)

def log_all_metrics(m):
    mlflow.log_metrics({
        "mae": m.mae, "mse": m.mse,
        "rmse": m.rmse, "r2": m.r2,
    })

def log_pred_plot(y, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, y_pred, alpha=0.5)
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Actual vs Predicted")
    mlflow.log_figure(fig, "pred_vs_actual.png")
    plt.close(fig)

def log_model_and_tags(model, X_train, X_test):
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_test.iloc[:3].copy()

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=MODEL_NAME,
        registered_model_name=REGISTERED_MODEL,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow>=2.15",
            "scikit-learn>=1.5",
            "pandas>=2.2",
            "numpy>=2.1",
        ],
    )

    mlflow.set_logged_model_tags(
        model_info.model_id,
        {"version": "v1", "scope": "demo"}
    )
    return model_info

# ---------------- Preprocessing ----------------
def encode_housing(df):
    df = df.copy()
    bin_map = {"yes": 1, "no": 0}
    df["mainroad"] = df["mainroad"].map(bin_map)
    df["guestroom"] = df["guestroom"].map(bin_map)
    df["basement"] = df["basement"].map(bin_map)
    df["hotwaterheating"] = df["hotwaterheating"].map(bin_map)
    df["airconditioning"] = df["airconditioning"].map(bin_map)
    df["prefarea"] = df["prefarea"].map(bin_map)
    df["furnishingstatus"] = df["furnishingstatus"].map({
        "furnished": 2, "semi-furnished": 1,
        "unfurnished": 0
    })
    return df

def remove_outliers(df):
    z = (df["price"] - df["price"].mean()) / df["price"].std()
    return df[np.abs(z) < 3]

# ---------------- Train Pipeline ----------------
def main():
    setup_mlflow(TRACKING_URI, EXPERIMENT_NAME)

    df = pd.read_csv(TRAIN_DATA_URL)
    df = encode_housing(df)
    df = remove_outliers(df)

    X = df[FEATURES_V1]
    y = df["price"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = Metrics(y, y_pred)

    with mlflow.start_run(run_name="train_v1"):
        log_params()
        log_all_metrics(metrics)
        log_pred_plot(y, y_pred)

        df.to_parquet("reference.parquet", index=False)
        mlflow.log_artifact("reference.parquet")

        log_model_and_tags(model, X, X)

        print("Model trained + registered successfully!")

if __name__ == "__main__":
    main()

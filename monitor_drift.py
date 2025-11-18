# monitor_drift.py

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score, mean_squared_error

TRACKING_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "housing_monitoring"

MODEL_NAME = "housing_price_model"

PROD_DATA_URL = "https://raw.githubusercontent.com/gchandra10/filestorage/refs/heads/main/Housing_Drifted.csv"

FEATURES_V1 = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus",
]

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
        "furnished": 2,
        "semi-furnished": 1,
        "unfurnished": 0,
    })
    return df

def remove_outliers(df):
    z = (df["price"] - df["price"].mean()) / df["price"].std()
    return df[np.abs(z) < 3]

def detect_numeric_drift(ref_df, prod_df, cols):
    records = []
    for col in cols:
        ref = ref_df[col]
        prod = prod_df[col]
        ref_mean = ref.mean()
        prod_mean = prod.mean()
        records.append({
            "feature": col,
            "ref_mean": ref_mean,
            "prod_mean": prod_mean,
            "mean_pct_change": (prod_mean - ref_mean) / ref_mean if ref_mean != 0 else np.nan,
            "ref_std": ref.std(),
            "prod_std": prod.std(),
            "std_ratio": prod.std() / ref.std() if ref.std() else np.nan,
        })
    return pd.DataFrame(records)

def evaluate_model_drift(model, ref_df, prod_df, feature_cols):
    X_ref = ref_df[feature_cols]
    y_ref = ref_df["price"]

    X_prod = prod_df[feature_cols]
    y_prod = prod_df["price"]

    y_ref_pred = model.predict(X_ref)
    y_prod_pred = model.predict(X_prod)

    return pd.DataFrame([
        {
            "dataset": "reference",
            "r2": r2_score(y_ref, y_ref_pred),
            "rmse": np.sqrt(mean_squared_error(y_ref, y_ref_pred)),
        },
        {
            "dataset": "production",
            "r2": r2_score(y_prod, y_prod_pred),
            "rmse": np.sqrt(mean_squared_error(y_prod, y_prod_pred)),
        },
    ])

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No registered versions for '{MODEL_NAME}'")

    latest = sorted(versions, key=lambda m: int(m.version))[-1]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")

    ref_path = mlflow.artifacts.download_artifacts(
        run_id=latest.run_id,
        artifact_path="reference.parquet",
    )
    ref_df = pd.read_parquet(ref_path)

    prod_df = pd.read_csv(PROD_DATA_URL)
    prod_df = encode_housing(prod_df)
    prod_df = remove_outliers(prod_df)

    numeric_drift = detect_numeric_drift(ref_df, prod_df, FEATURES_V1)
    perf_drift = evaluate_model_drift(model, ref_df, prod_df, FEATURES_V1)

    with mlflow.start_run(run_name="daily_monitoring"):
        mlflow.log_metric("r2_reference", perf_drift.loc[0, "r2"])
        mlflow.log_metric("r2_production", perf_drift.loc[1, "r2"])
        mlflow.log_metric("rmse_reference", perf_drift.loc[0, "rmse"])
        mlflow.log_metric("rmse_production", perf_drift.loc[1, "rmse"])

        numeric_drift.to_csv("numeric_drift.csv", index=False)
        mlflow.log_artifact("numeric_drift.csv")

        perf_drift.to_csv("performance_drift.csv", index=False)
        mlflow.log_artifact("performance_drift.csv")

        print(f"Monitoring completed for model version {latest.version}")
        print(perf_drift)

if __name__ == "__main__":
    main()

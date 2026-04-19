import os
import joblib
import pandas as pd
import logging
import yaml

import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import feature_engineering
from src.evaluation.evaluate import evaluate_clustering

from src.models.train_kmeans import train_kmeans, find_best_k
from src.models.train_gmm import train_gmm
from src.models.train_dbscan import train_dbscan
from src.models.train_hierarchical import train_hierarchical
from src.models.train_meanshift import train_meanshift


# =========================================================
# 🔹 MLflow Setup (FIXED)
# =========================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")   # 🔥 IMPORTANT
mlflow.set_experiment("Clustering_Comparison")


# =========================================================
# 🔹 LOAD CONFIG
# =========================================================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# =========================================================
# 🔹 LOGGING
# =========================================================
os.makedirs("artifacts/logs", exist_ok=True)

logging.basicConfig(
    filename="artifacts/logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# 🔹 SCALER
# =========================================================
def get_scaler():
    scaler_type = config["scaler"]["type"]

    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "power":
        return PowerTransformer(method="yeo-johnson")
    else:
        raise ValueError("Invalid scaler type")


# =========================================================
# 🔹 SAVE ARTIFACTS (LOCAL + MLFLOW)
# =========================================================
def save_artifacts(model_name, df, labels, X, metrics):

    base_path = config["artifacts"]["base_path"]
    model_path = os.path.join(base_path, model_name.lower())
    os.makedirs(model_path, exist_ok=True)

    # Clustered CSV
    df_copy = df.copy()
    df_copy["cluster"] = labels
    clustered_path = os.path.join(model_path, "clustered.csv")
    df_copy.to_csv(clustered_path, index=False)

    # Metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(model_path, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Plot
    plot_path = os.path.join(model_path, "clusters.png")
    try:
        pca_vis = PCA(n_components=2)
        X_vis = pca_vis.fit_transform(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap="viridis", s=10)
        plt.title(f"{model_name} Clusters (PCA)")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Plot error: {e}")

    # 🔥 Log to MLflow
    mlflow.log_artifact(clustered_path)
    mlflow.log_artifact(metrics_path)
    mlflow.log_artifact(plot_path)

    print(f"✅ Saved + Logged artifacts for {model_name}")


# =========================================================
# 🔹 SAVE MODEL
# =========================================================
def save_model_bundle(model_name, model, scaler, df, pca=None):

    path = f"models/{model_name.lower()}"
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, f"{path}/model.pkl")
    joblib.dump(scaler, f"{path}/scaler.pkl")
    joblib.dump(df.columns.tolist(), f"{path}/features.pkl")

    if pca is not None:
        joblib.dump(pca, f"{path}/pca.pkl")

    logging.info(f"{model_name} model saved")


# =========================================================
# 🔹 MAIN PIPELINE
# =========================================================
def main():

    # Load data
    df = load_data(config["data"]["raw_path"])
    df = preprocess_data(df)
    df = feature_engineering(df)

    # Save processed data
    os.makedirs(config["data"]["processed_path"], exist_ok=True)
    df.to_csv(config["data"]["processed_path"] + "featured_data.csv", index=False)

    # Scaling
    scaler = get_scaler()
    X = scaler.fit_transform(df)

    # PCA
    pca = None
    if config["pca"]["enabled"]:
        pca = PCA(n_components=config["pca"]["variance"])
        X = pca.fit_transform(X)

    results = []

    # =========================================================
    # 🔹 KMEANS
    # =========================================================
    with mlflow.start_run(run_name="KMeans"):

        best_k, _ = find_best_k(X, config["model"]["kmeans"]["k_range"])
        print("KMeans clusters:", best_k)

        model = train_kmeans(
            X,
            best_k,
            n_init=config["model"]["kmeans"]["n_init"],
            max_iter=config["model"]["kmeans"]["max_iter"]
        )

        labels = model.labels_
        metrics = evaluate_clustering(X, labels)

        mlflow.log_params({"model": "KMeans", "k": best_k})
        mlflow.log_metrics(metrics)

        save_model_bundle("kmeans", model, scaler, df, pca)
        save_artifacts("kmeans", df, labels, X, metrics)

        results.append(["KMeans", *metrics.values()])

    # =========================================================
    # 🔹 GMM
    # =========================================================
    with mlflow.start_run(run_name="GMM"):

        model, labels = train_gmm(X, best_k)
        metrics = evaluate_clustering(X, labels)

        mlflow.log_metrics(metrics)
        save_artifacts("gmm", df, labels, X, metrics)

        results.append(["GMM", *metrics.values()])

    # =========================================================
    # 🔹 DBSCAN
    # =========================================================
    with mlflow.start_run(run_name="DBSCAN"):

        model, labels = train_dbscan(X)
        metrics = evaluate_clustering(X, labels)

        mlflow.log_metrics(metrics)
        save_artifacts("dbscan", df, labels, X, metrics)

        results.append(["DBSCAN", *metrics.values()])

    # =========================================================
    # 🔹 HIERARCHICAL
    # =========================================================
    with mlflow.start_run(run_name="Hierarchical"):

        model, labels = train_hierarchical(X, best_k)
        metrics = evaluate_clustering(X, labels)

        mlflow.log_metrics(metrics)
        save_artifacts("hierarchical", df, labels, X, metrics)

        results.append(["Hierarchical", *metrics.values()])

    # =========================================================
    # 🔹 MEANSHIFT
    # =========================================================
    with mlflow.start_run(run_name="MeanShift"):

        model, labels = train_meanshift(X)
        metrics = evaluate_clustering(X, labels)

        mlflow.log_metrics(metrics)
        save_artifacts("meanshift", df, labels, X, metrics)

        results.append(["MeanShift", *metrics.values()])

    # =========================================================
    # 🔹 FINAL REPORT
    # =========================================================
    results_df = pd.DataFrame(
        results,
        columns=["Model", "Silhouette", "DB_Index", "CH_Score"]
    )

    os.makedirs("artifacts/reports", exist_ok=True)
    results_df.to_csv("artifacts/reports/model_comparison.csv", index=False)

    print("\n✅ FINAL MODEL COMPARISON:\n")
    print(results_df)


# =========================================================
# 🔹 RUN
# =========================================================
if __name__ == "__main__":
    main()
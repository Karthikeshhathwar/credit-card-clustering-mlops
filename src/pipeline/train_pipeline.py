import os
import joblib
import pandas as pd
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import feature_engineering
from src.evaluation.evaluate import evaluate_clustering

from src.models.train_kmeans import train_kmeans
from src.models.train_gmm import train_gmm
from src.models.train_dbscan import train_dbscan
from src.models.train_hierarchical import train_hierarchical
from src.models.train_meanshift import train_meanshift


# -------------------------------
# Logging Setup
# -------------------------------
os.makedirs("artifacts/logs", exist_ok=True)
logging.basicConfig(
    filename="artifacts/logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------
# Save Artifacts Function
# -------------------------------
def save_artifacts(model_name, df, labels, X, metrics):
    base_path = f"artifacts/{model_name.lower()}"
    os.makedirs(base_path, exist_ok=True)

    # Save clustered data
    df_copy = df.copy()
    df_copy["cluster"] = labels
    df_copy.to_csv(f"{base_path}/{model_name}_clustered.csv", index=False)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{base_path}/{model_name}_metrics.csv", index=False)

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f"{model_name} Clusters (PCA)")
    plt.savefig(f"{base_path}/{model_name}_clusters.png")
    plt.close()

    logging.info(f"{model_name} artifacts saved.")


# -------------------------------
# Save Model Function
# -------------------------------
def save_model(model_name, model, scaler=None):
    model_path = f"models/{model_name.lower()}"
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, f"{model_path}/model.pkl")

    # Save scaler ONLY for deployable models
    if scaler is not None:
        joblib.dump(scaler, f"{model_path}/scaler.pkl")

    logging.info(f"{model_name} model saved.")


# -------------------------------
# Main Pipeline
# -------------------------------
def main():

    DATA_PATH = "data/raw/credit_card.csv"

    mlflow.set_experiment("Clustering_Comparison")

    # -----------------------
    # Load & preprocess
    # -----------------------
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/preprocessed_data.csv", index=False)

    df = feature_engineering(df)
    df.to_csv("data/processed/featured_data.csv", index=False)

    # -----------------------
    # Scaling
    # -----------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    results_list = []

    # -----------------------
    # KMEANS (DEPLOYABLE)
    # -----------------------
    with mlflow.start_run(run_name="KMeans"):
        model = train_kmeans(X, 4)
        labels = model.labels_

        metrics = evaluate_clustering(X, labels)

        mlflow.log_params({"model": "KMeans", "n_clusters": 4})
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model")

        save_model("kmeans", model, scaler)
        save_artifacts("KMeans", df, labels, X, metrics)

        results_list.append(["KMeans", *metrics.values()])
        logging.info("KMeans completed.")

    # -----------------------
    # GMM (DEPLOYABLE)
    # -----------------------
    with mlflow.start_run(run_name="GMM"):
        model, labels = train_gmm(X, 4)

        metrics = evaluate_clustering(X, labels)

        mlflow.log_params({"model": "GMM", "n_components": 4})
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model")

        save_model("gmm", model, scaler)
        save_artifacts("GMM", df, labels, X, metrics)

        results_list.append(["GMM", *metrics.values()])
        logging.info("GMM completed.")

    # -----------------------
    # DBSCAN (ANALYSIS)
    # -----------------------
    with mlflow.start_run(run_name="DBSCAN"):
        model, labels = train_dbscan(X)

        metrics = evaluate_clustering(X, labels)

        mlflow.log_params({"model": "DBSCAN"})
        mlflow.log_metrics(metrics)

        save_model("dbscan", model)  # no scaler
        save_artifacts("DBSCAN", df, labels, X, metrics)

        results_list.append(["DBSCAN", *metrics.values()])
        logging.info("DBSCAN completed.")

    # -----------------------
    # Hierarchical (ANALYSIS)
    # -----------------------
    with mlflow.start_run(run_name="Hierarchical"):
        model, labels = train_hierarchical(X, 4)

        metrics = evaluate_clustering(X, labels)

        mlflow.log_params({"model": "Hierarchical"})
        mlflow.log_metrics(metrics)

        # Save only labels (no real model persistence)
        os.makedirs("models/hierarchical", exist_ok=True)
        joblib.dump(labels, "models/hierarchical/labels.pkl")

        save_artifacts("Hierarchical", df, labels, X, metrics)

        results_list.append(["Hierarchical", *metrics.values()])
        logging.info("Hierarchical completed.")

    # -----------------------
    # MeanShift (ANALYSIS)
    # -----------------------
    with mlflow.start_run(run_name="MeanShift"):
        model, labels = train_meanshift(X)

        metrics = evaluate_clustering(X, labels)

        mlflow.log_params({"model": "MeanShift"})
        mlflow.log_metrics(metrics)

        save_model("meanshift", model)
        save_artifacts("MeanShift", df, labels, X, metrics)

        results_list.append(["MeanShift", *metrics.values()])
        logging.info("MeanShift completed.")

    # -----------------------
    # Save comparison report
    # -----------------------
    results_df = pd.DataFrame(
        results_list,
        columns=["Model", "Silhouette", "DB_Index", "CH_Score"]
    )

    os.makedirs("artifacts/reports", exist_ok=True)
    results_df.to_csv("artifacts/reports/model_comparison.csv", index=False)

    print("\n✅ Model Comparison:\n")
    print(results_df)

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
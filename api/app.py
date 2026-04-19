import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from api.schema import CustomerData

# ✅ SAME preprocessing as training
from src.data.preprocess import preprocess_data


# =========================================================
# 🔹 Initialize FastAPI
# =========================================================
app = FastAPI(title="Credit Card Clustering API")


# =========================================================
# 🔹 Load Model Artifacts
# =========================================================
try:
    model = joblib.load("models/kmeans/model.pkl")
    scaler = joblib.load("models/kmeans/scaler.pkl")
    features = joblib.load("models/kmeans/features.pkl")

    # 🔥 Load PCA (critical fix)
    pca = joblib.load("models/kmeans/pca.pkl")

except Exception as e:
    raise RuntimeError(f"Error loading model artifacts: {e}")

n_clusters = model.n_clusters

# =========================================================
# 🔹 Cluster Interpretation
# =========================================================
def interpret_cluster(cluster, total_clusters):

    if total_clusters == 2:
        if cluster == 0:
            return {
                "segment": "Low Activity Customer",
                "description": "Low spending and engagement.",
                "recommendation": "Increase usage with offers."
            }
        else:
            return {
                "segment": "High Value Customer",
                "description": "High spending and active usage.",
                "recommendation": "Provide premium benefits."
            }

    elif total_clusters == 3:
        mapping = {
            0: "Low Activity",
            1: "Moderate User",
            2: "High Value"
        }
        return {
            "segment": mapping.get(cluster, "Unknown"),
            "description": "Cluster-based behavior",
            "recommendation": "Target accordingly"
        }
# =========================================================
# 🔹 Root Endpoint
# =========================================================
@app.get("/")
def home():
    return {"message": "Credit Card Clustering API is running 🚀"}


# =========================================================
# 🔹 Health Check
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# 🔹 Prediction Endpoint
# =========================================================
@app.post("/predict")
def predict(data: CustomerData):
    try:
        # -------------------------------
        # 1. Convert input → DataFrame
        # -------------------------------
        df = pd.DataFrame([data.dict()])

        # -------------------------------
        # 2. Apply SAME preprocessing
        # -------------------------------
        df = preprocess_data(df)

        # -------------------------------
        # 3. Align features EXACTLY
        # -------------------------------
        df = df.reindex(columns=features, fill_value=0)

        # -------------------------------
        # 4. Scale
        # -------------------------------
        X = scaler.transform(df)

        # -------------------------------
        # 5. Apply PCA (CRITICAL)
        # -------------------------------
        X = pca.transform(X)

        # -------------------------------
        # 6. Predict
        # -------------------------------
        cluster = int(model.predict(X)[0])

        # -------------------------------
        # 7. Interpret
        # -------------------------------
        result = interpret_cluster(cluster, n_clusters)

        return {
            "cluster": cluster,
            "segment": result["segment"],
            "description": result["description"],
            "recommendation": result["recommendation"]
        }

    except Exception as e:
        # Helpful debug response
        raise HTTPException(status_code=500, detail=str(e))
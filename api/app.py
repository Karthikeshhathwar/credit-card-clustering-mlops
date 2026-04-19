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
    pca = joblib.load("models/kmeans/pca.pkl")

except Exception as e:
    raise RuntimeError(f"Error loading model artifacts: {e}")


# Get cluster count dynamically
n_clusters = getattr(model, "n_clusters", 2)


# =========================================================
# 🔹 Cluster Interpretation (FIXED)
# =========================================================
def interpret_cluster(cluster, total_clusters):

    # Case: 2 clusters (your current model)
    if total_clusters == 2:
        if cluster == 0:
            return {
                "segment": "High Risk Customer",
                "description": "Frequent cash advance usage with low repayment behavior.",
                "recommendation": "Monitor closely, limit credit exposure, and offer repayment plans."
            }
        else:
            return {
                "segment": "High Value Customer",
                "description": "Active spender with strong repayment habits.",
                "recommendation": "Offer rewards, increase limits, and provide premium benefits."
            }

    # Case: 3 clusters (future-proof)
    elif total_clusters == 3:
        mapping = {
            0: ("Low Activity Customer", "Low spending and engagement."),
            1: ("Moderate User", "Balanced usage behavior."),
            2: ("High Value Customer", "High spending and strong repayment.")
        }

        segment, description = mapping.get(cluster, ("Unknown", "Unknown behavior"))

        return {
            "segment": segment,
            "description": description,
            "recommendation": "Apply targeted customer strategy."
        }

    # Default fallback
    return {
        "segment": "Unknown",
        "description": "Cluster not recognized.",
        "recommendation": "Further analysis required."
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
        # 5. Apply PCA
        # -------------------------------
        X = pca.transform(X)

        # -------------------------------
        # 6. Predict cluster
        # -------------------------------
        cluster = int(model.predict(X)[0])

        # -------------------------------
        # 7. Interpret cluster
        # -------------------------------
        result = interpret_cluster(cluster, n_clusters)

        return {
            "cluster": cluster,
            "segment": result["segment"],
            "description": result["description"],
            "recommendation": result["recommendation"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
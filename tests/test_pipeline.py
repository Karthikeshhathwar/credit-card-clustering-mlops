import os
import pandas as pd

def test_pipeline_runs():
    """
    Test if pipeline runs without crashing
    """
    from src.pipeline.train_pipeline import main

    try:
        main()
        assert True
    except Exception as e:
        assert False, f"Pipeline crashed: {e}"


def test_outputs_created():
    """
    Test if important output files are generated
    """

    # Expected files
    expected_files = [
        "artifacts/reports/model_comparison.csv",
        "models/kmeans/model.pkl",
        "models/kmeans/scaler.pkl",
        "models/kmeans/features.pkl"
    ]

    for file in expected_files:
        assert os.path.exists(file), f"Missing file: {file}"


def test_metrics_valid():
    """
    Test if metrics file has valid values
    """

    path = "artifacts/reports/model_comparison.csv"

    assert os.path.exists(path), "model_comparison.csv not found"

    df = pd.read_csv(path)

    # Check columns exist
    assert "Silhouette" in df.columns
    assert "DB_Index" in df.columns
    assert "CH_Score" in df.columns

    # Check values are not empty
    assert not df.empty, "Metrics dataframe is empty"

    # Check at least one good model
    assert df["Silhouette"].max() > 0, "All models have poor clustering"
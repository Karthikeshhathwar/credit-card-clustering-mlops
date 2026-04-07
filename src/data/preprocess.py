import numpy as np

def preprocess_data(df):
    df = df.copy()

    # Drop ID column
    if "CUST_ID" in df.columns:
        df.drop("CUST_ID", axis=1, inplace=True)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df
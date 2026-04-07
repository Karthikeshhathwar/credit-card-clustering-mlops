import numpy as np

def feature_engineering(df):
    df = df.copy()

    # Avoid division errors
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].replace(0, np.nan)
    df["BALANCE"] = df["BALANCE"].replace(0, np.nan)

    # Feature creation
    df["UTILIZATION"] = df["BALANCE"] / df["CREDIT_LIMIT"]
    df["SPENDING_RATIO"] = df["PURCHASES"] / df["CREDIT_LIMIT"]
    df["PAYMENT_RATIO"] = df["PAYMENTS"] / df["BALANCE"]
    df["CASH_ADV_RATIO"] = df["CASH_ADVANCE"] / df["CREDIT_LIMIT"]

    # Cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df
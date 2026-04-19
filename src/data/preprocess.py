import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------------------------------
    # 1. Drop ID column
    # -------------------------------
    if "CUST_ID" in df.columns:
        df.drop("CUST_ID", axis=1, inplace=True)

    # -------------------------------
    # 2. Fill missing values
    # -------------------------------
    df.fillna(df.median(numeric_only=True), inplace=True)

    # -------------------------------
    # 3. Log transform (handle skewness)
    # -------------------------------
    skew_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]
    for col in skew_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))  # ensure non-negative

    # -------------------------------
    # 4. Remove redundant / noisy features
    # -------------------------------
    df.drop(columns=[
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",

        # noisy frequency features
        "BALANCE_FREQUENCY",
        "PURCHASES_FREQUENCY",
        "ONEOFFPURCHASESFREQUENCY",
        "PURCHASESINSTALLMENTSFREQUENCY",
        "CASHADVANCEFREQUENCY"
    ], inplace=True, errors="ignore")

    # -------------------------------
    # 5. Feature Engineering (IMPORTANT)
    # -------------------------------
    eps = 1e-6  # avoid division by zero

    if "CREDIT_LIMIT" in df.columns:
        df["UTILIZATION"] = df["BALANCE"] / (df["CREDIT_LIMIT"] + eps)
        df["SPENDING_RATIO"] = df["PURCHASES"] / (df["CREDIT_LIMIT"] + eps)
        df["ADVANCE_RATIO"] = df["CASH_ADVANCE"] / (df["CREDIT_LIMIT"] + eps)

    if "BALANCE" in df.columns:
        df["PAYMENT_RATIO"] = df["PAYMENTS"] / (df["BALANCE"] + eps)

    # Behavioral features (HIGH IMPACT)
    if "PURCHASES_TRX" in df.columns:
        df["AVG_PURCHASE"] = df["PURCHASES"] / (df["PURCHASES_TRX"] + 1)

    if "CASHADVANCETRX" in df.columns:
        df["CASH_ADV_PER_TRX"] = df["CASH_ADVANCE"] / (df["CASHADVANCETRX"] + 1)

    if "MINIMUM_PAYMENTS" in df.columns:
        df["PAYMENT_COVERAGE"] = df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + 1)

    # -------------------------------
    # 6. Handle inf and NaN again
    # -------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # -------------------------------
    # 7. Outlier clipping (robust)
    # -------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    return df
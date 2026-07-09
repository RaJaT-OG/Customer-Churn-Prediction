"""
unified_dataset_builder.py
===========================
Builds ONE unified churn dataset by mapping E-Commerce, Banking, and Telco
customer data onto a shared set of GENERIC, INTERLINKABLE features.

Why not a raw merge?
---------------------
These three datasets have no shared customer key and almost no overlapping
raw columns (CreditScore vs MonthlyCharges vs CashbackAmount are not the
same thing). A naive row-concat would produce a dataset that is >70% NaN
per column and would let the model "cheat" by learning which industry a
row came from instead of learning real churn behaviour.

Instead, every dataset is mapped onto a small set of GENERIC features that
mean approximately the same thing in every business:

    tenure                  -> how long the customer has stayed
    engagement_score        -> normalised usage / activity intensity
    monetary_value          -> normalised spend / balance / charges
    product_or_service_count-> breadth of products/services used
    support_friction        -> complaints / support-ticket signal
    is_active               -> binary "currently engaged" flag
    gender                  -> demographic (kept, common to all 3)
    source_industry         -> which original dataset this came from
    churn                   -> unified binary target

Numeric features are min-max normalised WITHIN their own source dataset
BEFORE being stacked, so "Balance" (0-250,000) and "MonthlyCharges"
(0-120) end up on the same comparable 0-1 scale.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ════════════════════════════════════════════════════════════════════════════
# 1. PER-SOURCE LOADERS — clean each dataset to its native schema first
# ════════════════════════════════════════════════════════════════════════════


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max scale a numeric series to [0, 1], safe against zero-range."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)  # constant fallback
    return (s - lo) / (hi - lo)


def load_ecommerce(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="E Comm")

    out = pd.DataFrame()
    out["customer_ref"] = "ecom_" + df["CustomerID"].astype(str)
    out["tenure"] = df["Tenure"].fillna(df["Tenure"].median())
    out["gender"] = df["Gender"].str.strip().str.lower()

    # Engagement: app time + devices registered + order count, scaled then averaged
    engagement_raw = (
        _minmax(df["HourSpendOnApp"].fillna(0)) * 0.4
        + _minmax(df["NumberOfDeviceRegistered"].fillna(0)) * 0.3
        + _minmax(df["OrderCount"].fillna(0)) * 0.3
    )
    out["engagement_score"] = engagement_raw

    # Monetary proxy: cashback amount (spend-correlated)
    out["monetary_value"] = _minmax(df["CashbackAmount"].fillna(0))

    out["product_or_service_count"] = df["OrderCount"].fillna(0)
    out["support_friction"] = df["Complain"].fillna(0)
    out["is_active"] = (df["DaySinceLastOrder"].fillna(99) <= 30).astype(int)
    out["source_industry"] = "ecommerce"
    out["churn"] = df["Churn"].astype(int)

    return out


def load_banking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["customer_ref"] = "bank_" + df["CustomerId"].astype(str)
    out["tenure"] = df["Tenure"].fillna(df["Tenure"].median())
    out["gender"] = df["Gender"].str.strip().str.lower()

    # Engagement: active member flag + product count, scaled then averaged
    engagement_raw = (
        df["IsActiveMember"].fillna(0) * 0.5
        + _minmax(df["NumOfProducts"].fillna(0)) * 0.5
    )
    out["engagement_score"] = engagement_raw

    out["monetary_value"] = _minmax(df["Balance"].fillna(0))
    out["product_or_service_count"] = df["NumOfProducts"].fillna(0)
    out["support_friction"] = 0  # no equivalent column -> assume none reported
    out["is_active"] = df["IsActiveMember"].fillna(0).astype(int)
    out["source_industry"] = "banking"
    out["churn"] = df["Exited"].astype(int)

    return out


def load_telco(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    service_count = df[service_cols].apply(
        lambda row: sum(str(v).strip().lower() == "yes" for v in row), axis=1
    )

    out = pd.DataFrame()
    out["customer_ref"] = "telco_" + df["customerID"].astype(str)
    out["tenure"] = df["tenure"].fillna(df["tenure"].median())
    out["gender"] = df["gender"].str.strip().str.lower()

    # Engagement: service breadth + paperless/contract engagement proxy
    engagement_raw = (
        _minmax(service_count) * 0.6 + _minmax(df["tenure"].fillna(0)) * 0.4
    )
    out["engagement_score"] = engagement_raw

    out["monetary_value"] = _minmax(df["MonthlyCharges"].fillna(0))
    out["product_or_service_count"] = service_count
    out["support_friction"] = (
        df["TechSupport"].str.strip().str.lower() == "no"
    ).astype(int)
    out["is_active"] = (df["Contract"] != "Month-to-month").astype(int)
    out["source_industry"] = "telco"
    out["churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. BUILD UNIFIED DATASET
# ════════════════════════════════════════════════════════════════════════════

GENERIC_COLUMNS = [
    "customer_ref",
    "source_industry",
    "tenure",
    "gender",
    "engagement_score",
    "monetary_value",
    "product_or_service_count",
    "support_friction",
    "is_active",
    "churn",
]


def build_unified_dataset(
    ecommerce_path: Path = DATA_DIR / "E_Commerce_Dataset.xlsx",
    banking_path: Path = DATA_DIR / "Churn_Modelling.csv",
    telco_path: Path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    """
    Load all three sources, map to the generic schema, and stack them
    into one unified, model-ready DataFrame.
    """
    print("[INFO] Loading and mapping E-Commerce dataset …")
    df_ecom = load_ecommerce(ecommerce_path)

    print("[INFO] Loading and mapping Banking dataset …")
    df_bank = load_banking(banking_path)

    print("[INFO] Loading and mapping Telco dataset …")
    df_telco = load_telco(telco_path)

    unified = pd.concat([df_ecom, df_bank, df_telco], ignore_index=True)
    unified = unified[GENERIC_COLUMNS]

    # Tenure is on different native scales (months vs years) -> normalise per source
    # then keep a single global 0-1 tenure_score for modelling, while keeping
    # raw tenure for interpretability.
    unified["tenure_score"] = unified.groupby("source_industry")["tenure"].transform(
        lambda s: _minmax(s)
    )

    print(
        f"\n[INFO] Unified dataset built: {unified.shape[0]:,} rows × {unified.shape[1]} cols"
    )
    print(
        f"[INFO] Source breakdown:\n{unified['source_industry'].value_counts().to_string()}"
    )
    print(f"[INFO] Overall churn rate: {unified['churn'].mean():.2%}")
    print(
        f"[INFO] Churn rate by source:\n{unified.groupby('source_industry')['churn'].mean().round(4).to_string()}"
    )

    return unified


if __name__ == "__main__":
    unified_df = build_unified_dataset()
    out_path = DATA_DIR / "unified_churn_dataset.csv"
    unified_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved -> {out_path}")
    print("\nPreview:")
    print(unified_df.head(10).to_string(index=False))

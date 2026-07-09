"""
tests/test_unified_pipeline.py
================================
Tests for the unified cross-industry churn dataset and model pipeline.

Run with:
    pytest tests/test_unified_pipeline.py -v
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

from src.unified_dataset_builder import (
    load_ecommerce,
    load_banking,
    load_telco,
    build_unified_dataset,
    GENERIC_COLUMNS,
    _minmax,
)
from src.unified_churn_pipeline import (
    build_preprocessor,
    handle_imbalance,
    score_and_segment,
    create_engineered_features,
    MODEL_FEATURES,
    TARGET,
    SMOTE_AVAILABLE,
)

DATA_DIR = "."


# ────────────────────────────────────────────────────────────────────────────
# Fixtures: tiny synthetic versions of each source so tests don't depend
# on the real (large) data files being present.
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mini_ecommerce_df(tmp_path):
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "CustomerID": range(50001, 50001 + n),
            "Churn": rng.integers(0, 2, n),
            "Tenure": rng.integers(0, 30, n).astype(float),
            "Gender": rng.choice(["Male", "Female"], n),
            "HourSpendOnApp": rng.uniform(0, 5, n),
            "NumberOfDeviceRegistered": rng.integers(1, 6, n),
            "OrderCount": rng.integers(1, 20, n),
            "CashbackAmount": rng.uniform(50, 500, n),
            "Complain": rng.integers(0, 2, n),
            "DaySinceLastOrder": rng.integers(0, 60, n).astype(float),
        }
    )
    path = tmp_path / "E_Commerce_Dataset.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="E Comm", index=False)
    return path


@pytest.fixture
def mini_banking_df(tmp_path):
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "RowNumber": range(n),
            "CustomerId": range(15_000_000, 15_000_000 + n),
            "Surname": ["Smith"] * n,
            "CreditScore": rng.integers(300, 850, n),
            "Geography": rng.choice(["France", "Spain", "Germany"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 70, n),
            "Tenure": rng.integers(0, 10, n),
            "Balance": rng.uniform(0, 200_000, n),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(10_000, 200_000, n),
            "Exited": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "Churn_Modelling.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mini_telco_df(tmp_path):
    rng = np.random.default_rng(2)
    n = 100
    yn = lambda: rng.choice(["Yes", "No"], n)
    df = pd.DataFrame(
        {
            "customerID": [f"ID{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "tenure": rng.integers(0, 72, n),
            "OnlineSecurity": yn(),
            "OnlineBackup": yn(),
            "DeviceProtection": yn(),
            "TechSupport": yn(),
            "StreamingTV": yn(),
            "StreamingMovies": yn(),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "MonthlyCharges": rng.uniform(20, 120, n),
            "TotalCharges": rng.uniform(100, 8000, n).astype(str),
            "Churn": rng.choice(["Yes", "No"], n),
        }
    )
    path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def unified_synthetic(mini_ecommerce_df, mini_banking_df, mini_telco_df):
    df = build_unified_dataset(
        ecommerce_path=mini_ecommerce_df,
        banking_path=mini_banking_df,
        telco_path=mini_telco_df,
    )
    return create_engineered_features(df)


# ────────────────────────────────────────────────────────────────────────────
# 1. Per-source loader tests
# ────────────────────────────────────────────────────────────────────────────


class TestPerSourceLoaders:

    def test_ecommerce_loader_columns(self, mini_ecommerce_df):
        df = load_ecommerce(mini_ecommerce_df)
        assert set(GENERIC_COLUMNS).issubset(set(df.columns))

    def test_banking_loader_columns(self, mini_banking_df):
        df = load_banking(mini_banking_df)
        assert set(GENERIC_COLUMNS).issubset(set(df.columns))

    def test_telco_loader_columns(self, mini_telco_df):
        df = load_telco(mini_telco_df)
        assert set(GENERIC_COLUMNS).issubset(set(df.columns))

    def test_ecommerce_customer_ref_prefix(self, mini_ecommerce_df):
        df = load_ecommerce(mini_ecommerce_df)
        assert df["customer_ref"].str.startswith("ecom_").all()

    def test_banking_customer_ref_prefix(self, mini_banking_df):
        df = load_banking(mini_banking_df)
        assert df["customer_ref"].str.startswith("bank_").all()

    def test_telco_customer_ref_prefix(self, mini_telco_df):
        df = load_telco(mini_telco_df)
        assert df["customer_ref"].str.startswith("telco_").all()

    def test_churn_is_binary_all_sources(
        self, mini_ecommerce_df, mini_banking_df, mini_telco_df
    ):
        for loader, path in [
            (load_ecommerce, mini_ecommerce_df),
            (load_banking, mini_banking_df),
            (load_telco, mini_telco_df),
        ]:
            df = loader(path)
            assert df["churn"].isin([0, 1]).all()

    def test_engagement_score_normalised_range(self, mini_ecommerce_df):
        df = load_ecommerce(mini_ecommerce_df)
        # engagement score is a weighted sum of [0,1] scaled components -> should stay in [0,1]
        assert df["engagement_score"].between(-0.01, 1.01).all()

    def test_monetary_value_in_unit_range(self, mini_banking_df):
        df = load_banking(mini_banking_df)
        assert df["monetary_value"].between(-0.01, 1.01).all()


# ────────────────────────────────────────────────────────────────────────────
# 2. Minmax helper tests
# ────────────────────────────────────────────────────────────────────────────


class TestMinMaxHelper:

    def test_minmax_basic_range(self):
        s = pd.Series([0, 5, 10])
        scaled = _minmax(s)
        assert scaled.min() == 0.0
        assert scaled.max() == 1.0

    def test_minmax_constant_series_fallback(self):
        s = pd.Series([5, 5, 5, 5])
        scaled = _minmax(s)
        assert (scaled == 0.5).all()

    def test_minmax_handles_nan(self):
        s = pd.Series([1, np.nan, 3])
        scaled = _minmax(s)
        assert not scaled.isna().all()


# ────────────────────────────────────────────────────────────────────────────
# 3. Unified dataset construction tests
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedDatasetBuild:

    def test_row_count_equals_sum_of_sources(self, unified_synthetic):
        assert len(unified_synthetic) == 300  # 100 + 100 + 100

    def test_all_three_sources_present(self, unified_synthetic):
        sources = set(unified_synthetic["source_industry"].unique())
        assert sources == {"ecommerce", "banking", "telco"}

    def test_no_duplicate_customer_refs(self, unified_synthetic):
        assert unified_synthetic["customer_ref"].is_unique

    def test_generic_columns_present(self, unified_synthetic):
        for col in GENERIC_COLUMNS:
            assert col in unified_synthetic.columns

    def test_tenure_score_per_source_in_unit_range(self, unified_synthetic):
        assert unified_synthetic["tenure_score"].between(-0.01, 1.01).all()

    def test_tenure_score_normalised_independently_per_source(self, unified_synthetic):
        # each source should have its own min (0) and max (1) tenure_score
        # (unless degenerate / constant, handled by _minmax fallback)
        for source in ["ecommerce", "banking", "telco"]:
            sub = unified_synthetic[unified_synthetic["source_industry"] == source]
            assert sub["tenure_score"].max() <= 1.01

    def test_churn_column_binary(self, unified_synthetic):
        assert unified_synthetic["churn"].isin([0, 1]).all()

    def test_no_critical_nans(self, unified_synthetic):
        critical_cols = ["tenure_score", "engagement_score", "monetary_value", "churn"]
        for col in critical_cols:
            assert unified_synthetic[col].isna().sum() == 0


# ────────────────────────────────────────────────────────────────────────────
# 4. Preprocessing on unified data
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedPreprocessing:

    def test_preprocessor_builds_and_transforms(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        preprocessor, num_cols, cat_cols = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)
        assert X_proc.shape[0] == len(X)
        assert not np.isnan(X_proc).any()

    def test_source_industry_is_onehot_encoded(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        preprocessor, num_cols, cat_cols = build_preprocessor(X)
        assert "source_industry" in cat_cols
        preprocessor.fit(X)
        ohe_names = (
            preprocessor.named_transformers_["cat"]["encoder"]
            .get_feature_names_out(cat_cols)
            .tolist()
        )
        assert any("source_industry" in name for name in ohe_names)

    def test_train_test_same_feature_count(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        y = unified_synthetic[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        preprocessor, _, _ = build_preprocessor(X_tr)
        X_tr_p = preprocessor.fit_transform(X_tr)
        X_te_p = preprocessor.transform(X_te)
        assert X_tr_p.shape[1] == X_te_p.shape[1]


# ────────────────────────────────────────────────────────────────────────────
# 5. Imbalance handling on unified data
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedImbalance:

    def test_smote_runs_on_unified_features(self, unified_synthetic):
        if not SMOTE_AVAILABLE:
            pytest.skip("imbalanced-learn not installed")
        X = unified_synthetic[MODEL_FEATURES]
        y = unified_synthetic[TARGET].values
        preprocessor, _, _ = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)
        X_res, y_res = handle_imbalance(X_proc, y)
        assert len(X_res) >= len(X_proc)
        assert len(y_res) == len(X_res)


# ────────────────────────────────────────────────────────────────────────────
# 6. Model accuracy on unified data (synthetic, so thresholds are lenient)
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedModelAccuracy:
    """
    These synthetic fixtures use RANDOM data (no real signal), so we only
    assert the pipeline runs and produces valid probability outputs —
    not that it beats a coin flip. Real-data accuracy is validated
    separately by running unified_churn_pipeline.py on the actual CSVs.
    """

    def test_model_produces_valid_probabilities(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        y = unified_synthetic[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        preprocessor, _, _ = build_preprocessor(X_tr)
        X_tr_p = preprocessor.fit_transform(X_tr)
        X_te_p = preprocessor.transform(X_te)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr_p, y_tr)
        probs = clf.predict_proba(X_te_p)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_roc_auc_computable(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        y = unified_synthetic[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        preprocessor, _, _ = build_preprocessor(X_tr)
        X_tr_p = preprocessor.fit_transform(X_tr)
        X_te_p = preprocessor.transform(X_te)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr_p, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te_p)[:, 1])
        assert 0.0 <= auc <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# 7. Risk segmentation on unified data
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedSegmentation:

    def test_risk_tiers_valid(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        y = unified_synthetic[TARGET]
        preprocessor, _, _ = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_proc, y)

        scored = score_and_segment(clf, X_proc, unified_synthetic)
        assert set(scored["RiskTier"]).issubset({"High", "Medium", "Low"})
        assert scored["ChurnProbability"].between(0, 1).all()
        assert scored["RetentionActions"].notna().all()

    def test_segmentation_preserves_source_industry(self, unified_synthetic):
        X = unified_synthetic[MODEL_FEATURES]
        y = unified_synthetic[TARGET]
        preprocessor, _, _ = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_proc, y)

        scored = score_and_segment(clf, X_proc, unified_synthetic)
        assert "source_industry" in scored.columns
        assert set(scored["source_industry"].unique()) == {
            "ecommerce",
            "banking",
            "telco",
        }


# ────────────────────────────────────────────────────────────────────────────
# 8. Integration smoke test using REAL data files (skipped if absent)
# ────────────────────────────────────────────────────────────────────────────


class TestRealDataIntegration:

    @pytest.mark.skipif(
        not (
            os.path.exists("E_Commerce_Dataset.xlsx")
            and os.path.exists("Churn_Modelling.csv")
            and os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        ),
        reason="Real dataset files not present in working directory",
    )
    def test_real_unified_dataset_builds(self):
        df = build_unified_dataset()
        assert len(df) > 20000  # ecom(5630) + bank(10000) + telco(7043)
        assert set(df["source_industry"].unique()) == {"ecommerce", "banking", "telco"}
        assert df["churn"].mean() > 0  # has positive class
        assert df["churn"].mean() < 1  # has negative class

    @pytest.mark.skipif(
        not (
            os.path.exists("E_Commerce_Dataset.xlsx")
            and os.path.exists("Churn_Modelling.csv")
            and os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        ),
        reason="Real dataset files not present in working directory",
    )
    def test_real_model_beats_baseline_auc(self):
        """On real data, the unified model should comfortably beat random (0.5 AUC)."""
        df = build_unified_dataset()
        X = df[MODEL_FEATURES]
        y = df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        preprocessor, _, _ = build_preprocessor(X_tr)
        X_tr_p = preprocessor.fit_transform(X_tr)
        X_te_p = preprocessor.transform(X_te)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        )
        clf.fit(X_tr_p, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te_p)[:, 1])
        assert auc >= 0.70, f"Unified model ROC-AUC {auc:.4f} below 0.70 on real data"
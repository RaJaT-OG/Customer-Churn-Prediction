"""
unified_churn_pipeline.py
==========================
Trains a churn model on the UNIFIED dataset (built by unified_dataset_builder.py)
which combines E-Commerce, Banking, and Telco customers into one schema using
generic, cross-industry features.

Pipeline: Unified Data -> Preprocessing -> Imbalance Handling (SMOTE) ->
          Model Training & Evaluation -> Risk Segmentation -> SHAP ->
          Retention Recommendations
"""

import joblib
from src.logger import logger
import warnings

warnings.filterwarnings("ignore")
from src.config_loader import load_config
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ── MLflow ────────────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] mlflow not installed. pip install mlflow")

config = load_config()

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed. pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] xgboost not installed. pip install xgboost")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed. pip install shap")

from src.unified_dataset_builder import (
    build_unified_dataset,
    GENERIC_COLUMNS,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# ── MLflow tracking URI — SQLite works on all platforms including Windows ──
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlruns.db'}"
MLFLOW_EXPERIMENT = "churn-prediction"


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD UNIFIED DATA
# ════════════════════════════════════════════════════════════════════════════


def load_or_build_unified(
    path: Path = DATA_DIR / "unified_churn_dataset.csv",
) -> pd.DataFrame:
    if path.exists():
        logger.info(f"Loading existing unified dataset from {path}")
        return pd.read_csv(path)
    logger.info("Unified dataset not found on disk — building it now...")
    df = build_unified_dataset()
    df.to_csv(path, index=False)
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURE SET FOR MODELLING
# ════════════════════════════════════════════════════════════════════════════

MODEL_FEATURES = [
    "tenure_score",
    "gender",
    "engagement_score",
    "monetary_value",
    "product_or_service_count",
    "support_friction",
    "is_active",
    "source_industry",
    "customer_value_score",
    "support_intensity",
    "engagement_per_tenure",
    "value_per_product",
    "active_high_value",
    "tenure_engagement",
    "value_segment",
    "tenure_segment",
    "engagement_segment",
]
TARGET = "churn"


def _safe_qcut(series: pd.Series, q: int, labels: list) -> pd.Series:
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except ValueError:
        pass
    for n_bins in range(min(q, series.nunique()), 1, -1):
        try:
            return pd.qcut(series, q=n_bins, labels=labels[:n_bins], duplicates="drop")
        except ValueError:
            continue
    return pd.Series(["Mixed"] * len(series), index=series.index)


def create_engineered_features(df):
    df = df.copy()
    eps = 1e-6

    df["customer_value_score"] = df["engagement_score"] * df["monetary_value"]
    df["support_intensity"] = df["support_friction"] / (
        df["product_or_service_count"] + eps
    )
    df["engagement_per_tenure"] = df["engagement_score"] / (df["tenure_score"] + eps)
    df["value_per_product"] = df["monetary_value"] / (
        df["product_or_service_count"] + eps
    )
    df["active_high_value"] = df["is_active"] * df["monetary_value"]
    df["tenure_engagement"] = df["tenure_score"] * df["engagement_score"]

    df["value_segment"] = _safe_qcut(
        df["monetary_value"], q=4, labels=["Low", "Medium", "High", "Premium"]
    )
    df["tenure_segment"] = _safe_qcut(
        df["tenure_score"], q=4, labels=["New", "Growing", "Established", "Loyal"]
    )
    df["engagement_segment"] = _safe_qcut(
        df["engagement_score"], q=4, labels=["Low", "Medium", "High", "Very High"]
    )

    return df


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


# ════════════════════════════════════════════════════════════════════════════
# 3. IMBALANCE HANDLING
# ════════════════════════════════════════════════════════════════════════════


def handle_imbalance(X_train, y_train):
    churn_rate = y_train.mean()
    logger.info(f"Training churn rate: {churn_rate:.2%}")
    if churn_rate < 0.3 and SMOTE_AVAILABLE:
        X_res, y_res = SMOTE(random_state=config["smote"]["random_state"]).fit_resample(
            X_train, y_train
        )
        logger.info(f"SMOTE applied: {len(y_train):,} -> {len(y_res):,} samples")
        return X_res, y_res
    return X_train, y_train


# ════════════════════════════════════════════════════════════════════════════
# 4. MODELS
# ════════════════════════════════════════════════════════════════════════════


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
            class_weight=config["logistic_regression"].get("class_weight"),
            random_state=config["data"]["random_state"],
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=config["random_forest"]["n_estimators"],
            max_depth=config["random_forest"]["max_depth"],
            class_weight=config["random_forest"]["class_weight"],
            random_state=config["data"]["random_state"],
            n_jobs=-1,
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=config["xgboost"]["n_estimators"],
            learning_rate=config["xgboost"]["learning_rate"],
            max_depth=config["xgboost"]["max_depth"],
            random_state=config["data"]["random_state"],
            scale_pos_weight=config["xgboost"]["scale_pos_weight"],
        )
    return models


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n{'-'*50}\n  {name}\n{'-'*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    return metrics


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train all models, log each run to MLflow, return best."""

    # ── Setup MLflow ───────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        print(f"\n[MLflow] Tracking URI : {MLFLOW_TRACKING_URI}")
        print(f"[MLflow] Experiment   : {MLFLOW_EXPERIMENT}")

    results, best_auc, best_model, best_name = {}, 0.0, None, ""

    for name, clf in get_models().items():
        logger.info(f"Training {name}")

        # ── MLflow run per model ───────────────────────────────────────────
        if MLFLOW_AVAILABLE:
            run = mlflow.start_run(run_name=name)

        clf.fit(X_train, y_train)
        metrics = evaluate_model(name, clf, X_test, y_test)

        # ── Log to MLflow ──────────────────────────────────────────────────
        if MLFLOW_AVAILABLE:
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("smote_used", SMOTE_AVAILABLE)

            # Log hyperparameters
            try:
                for param, val in clf.get_params().items():
                    mlflow.log_param(param, val)
            except Exception:
                pass

            # Log metrics
            mlflow.log_metric("accuracy", metrics["Accuracy"])
            mlflow.log_metric("precision", metrics["Precision"])
            mlflow.log_metric("recall", metrics["Recall"])
            mlflow.log_metric("f1_score", metrics["F1"])
            mlflow.log_metric("roc_auc", metrics["ROC-AUC"])

            # Log model artifact — use correct flavor per model type
            model_reg_name = f"churn-{name.lower().replace(' ', '-')}"
            try:
                if XGB_AVAILABLE and isinstance(clf, XGBClassifier):
                    mlflow.xgboost.log_model(
                        clf,
                        artifact_path="model",
                        registered_model_name=model_reg_name,
                    )
                else:
                    mlflow.sklearn.log_model(
                        clf,
                        artifact_path="model",
                        registered_model_name=model_reg_name,
                    )
            except Exception as log_err:
                print(f"[MLflow] Model artifact logging skipped: {log_err}")

            mlflow.end_run()
            print(f"[MLflow] Logged run for {name} — AUC={metrics['ROC-AUC']:.4f}")

        results[name] = {"model": clf, "metrics": metrics}

        if metrics["ROC-AUC"] > best_auc:
            best_auc = metrics["ROC-AUC"]
            best_model = clf
            best_name = name

    print(f"\n[INFO] Best model: {best_name} (ROC-AUC = {best_auc:.4f})")

    # ── Promote best model to Production in MLflow registry ───────────────
    if MLFLOW_AVAILABLE:
        try:
            client = MlflowClient(MLFLOW_TRACKING_URI)
            model_name = f"churn-{best_name.lower().replace(' ', '-')}"
            versions = client.get_latest_versions(model_name, stages=["None"])
            if versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=versions[0].version,
                    stage="Production",
                )
                print(f"[MLflow] {model_name} v{versions[0].version} → Production ✅")
        except Exception as e:
            print(f"[MLflow] Registry promotion skipped: {e}")

    return results, best_model, best_name


# ════════════════════════════════════════════════════════════════════════════
# 5. RISK SEGMENTATION + RETENTION
# ════════════════════════════════════════════════════════════════════════════

RETENTION_RULES = {
    "High": [
        "Immediate personal outreach (call or dedicated rep).",
        "Offer tailored discount / fee waiver / loyalty reward.",
        "Escalate any open support issue same-day.",
    ],
    "Medium": [
        "Targeted re-engagement email or push notification.",
        "Highlight underused product/service features.",
        "Offer a modest incentive to increase engagement.",
    ],
    "Low": [
        "Include in standard newsletter / loyalty programme.",
        "Periodic check-in at natural milestones.",
    ],
}


def score_and_segment(model, X_proc, original_df: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(X_proc)[:, 1]
    scored = original_df.copy().reset_index(drop=True)
    scored["ChurnProbability"] = probs.round(4)

    def tier(p):
        if p >= config["risk_thresholds"]["high"]:
            return "High"
        if p >= config["risk_thresholds"]["medium"]:
            return "Medium"
        return "Low"

    scored["RiskTier"] = scored["ChurnProbability"].apply(tier)
    scored["RetentionActions"] = scored["RiskTier"].map(
        lambda t: " | ".join(RETENTION_RULES.get(t, ["No action required."]))
    )

    print("\n[INFO] Risk Tier Distribution:")
    print(scored["RiskTier"].value_counts().to_string())
    print("\n[INFO] Risk Tier by source industry:")
    print(pd.crosstab(scored["source_industry"], scored["RiskTier"]).to_string())
    return scored


# ════════════════════════════════════════════════════════════════════════════
# 6. SHAP
# ════════════════════════════════════════════════════════════════════════════


def explain_with_shap(model, X_sample, feature_names, top_n=10):
    if not SHAP_AVAILABLE:
        print("[WARN] SHAP not available — skipping.")
        return None

    print(f"\n[SHAP] Computing explanations on {X_sample.shape[0]} samples …")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        background = shap.sample(X_sample, min(100, X_sample.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample[:50])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1).ravel()
    mean_abs = mean_abs.ravel()

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Mean|SHAP|": mean_abs})
        .sort_values("Mean|SHAP|", ascending=False)
        .head(top_n)
    )

    print(f"\n[SHAP] Top {top_n} Cross-Industry Churn Drivers:")
    print(importance_df.to_string(index=False))
    return importance_df


# ════════════════════════════════════════════════════════════════════════════
# 7. ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════


def run_unified_pipeline():
    print("\n" + "=" * 60)
    print("  UNIFIED CROSS-INDUSTRY CHURN PREDICTION")
    print("=" * 60)

    df = load_or_build_unified()
    df = create_engineered_features(df)

    X = df[MODEL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    print(f"\n[INFO] Train: {len(X_train):,}  Test: {len(X_test):,}")

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    ohe_names = (
        preprocessor.named_transformers_["cat"]["encoder"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    feature_names = num_cols + ohe_names

    X_train_bal, y_train_bal = handle_imbalance(X_train_proc, y_train.values)

    results, best_model, best_name = train_and_evaluate(
        X_train_bal, X_test_proc, y_train_bal, y_test.values
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "unified_best_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "unified_preprocessor.pkl")
    print(f"[INFO] Model saved -> {MODEL_DIR / 'unified_best_model.pkl'}")

    X_all_proc = preprocessor.transform(X)
    scored_df = score_and_segment(best_model, X_all_proc, df)

    shap_sample_size = min(config["shap"]["sample_size"], X_test_proc.shape[0])
    shap_df = explain_with_shap(
        best_model, X_test_proc[:shap_sample_size], feature_names
    )

    out_path = DATA_DIR / "unified_churn_scored.csv"
    scored_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Scored output saved -> {out_path}")

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (UNIFIED DATASET)")
    print("=" * 60)
    summary = pd.DataFrame([v["metrics"] for v in results.values()])
    print(summary.to_string(index=False))

    if MLFLOW_AVAILABLE:
        print(f"\n[MLflow] All runs saved to: {MLFLOW_TRACKING_URI}")
        print(f"[MLflow] Open Model Health page in Streamlit to view them.")

    return {
        "scored_df": scored_df,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "results": results,
        "shap_df": shap_df,
    }


if __name__ == "__main__":
    run_unified_pipeline()

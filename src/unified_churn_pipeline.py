"""
unified_churn_pipeline.py
==========================
Trains a churn model on the UNIFIED dataset (built by unified_dataset_builder.py)
which combines E-Commerce, Banking, and Telco customers into one schema using
generic, cross-industry features.

Pipeline: Unified Data -> Preprocessing -> Imbalance Handling (SMOTE) ->
          Model Training & Evaluation -> Calibration -> Risk Segmentation ->
          SHAP -> Retention Recommendations

NOTE ON CALIBRATION
--------------------
SMOTE rebalances the *training* split to ~50/50 churn/no-churn so the model
can learn the minority class properly. Left uncorrected, that also shifts
the model's predicted probabilities upward relative to the real-world churn
rate (e.g. avg predicted churn prob >> actual churn rate), because the model
was fit against an artificial 50% prior.

To fix this we carve out a THIRD split — a calibration set — that is never
touched by SMOTE (kept at the real class distribution), and use it to
recalibrate predict_proba() via CalibratedClassifierCV (isotonic). The test
set also stays at the real distribution, so evaluation metrics remain honest.

Because CalibratedClassifierCV wraps the underlying model, it no longer
exposes tree internals needed by SHAP's TreeExplainer or .feature_importances_.
We therefore save TWO model files:
  - unified_best_model.pkl  -> calibrated model, used for predict_proba() / risk tiers
  - unified_raw_model.pkl   -> uncalibrated model, used for SHAP + feature_importances_
Isotonic calibration is monotonic, so SHAP direction/ranking on the raw model
still reflects what drives the calibrated predictions.

NOTE ON SEGMENT BINS (value/tenure/engagement_segment)
-------------------------------------------------------
These three columns used to be built with pd.qcut() called fresh on
whatever DataFrame happened to be passed to create_engineered_features().
That meant:
  - calibration/test splits could get slightly different bin edges than
    training (mild leakage / train-serve skew)
  - a single-customer row (app.py "Single Customer" page) can't form
    quartiles at all and silently fell back to a meaningless "Mixed"
    bucket every time
Fix: fit_segment_bins() computes the bin edges ONCE, on the training
split only, right after the raw train/calib/test split and BEFORE any
engineered features exist. Those edges are saved as
unified_segment_bins.pkl and reapplied everywhere else (calibration set,
test set, full-dataset scoring, and every app.py prediction) via
apply_segment_bins() / create_engineered_features(df, segment_bins=...).
The old qcut-per-call behavior is kept ONLY as a fallback for ad-hoc
exploration when no segment_bins are supplied — never for calibration,
test scoring, or production inference.
"""

import joblib
from src.logger import logger
import warnings

warnings.filterwarnings("ignore")
from src.config_loader import load_config
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ── Silence MLflow/Alembic/SQLAlchemy setup chatter ────────────────────────
# A plain logging.getLogger("alembic").setLevel(WARNING) isn't enough here:
# MLflow's SQLAlchemy store re-checks the DB schema on tracking calls
# (log_param/log_metric/log_model — and this pipeline calls log_param once
# per hyperparameter, across 3 models, so that's 80-100+ calls), and each
# check re-runs Alembic's own fileConfig(), which resets alembic's logger
# level back to INFO — silently undoing our .setLevel() call.
# logging.disable() sits BELOW per-logger level checks (a global floor in
# the logging module), so nothing alembic does to its own logger can
# re-enable these messages.
logging.disable(logging.INFO)

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
from src.calibration import CalibratedModel
from sklearn.isotonic import IsotonicRegression

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
    path: Path = DATA_DIR / "unified_churn_dataset (1).csv",
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

# Which raw column each segment is derived from, and its label set.
SEGMENT_SPECS = {
    "value_segment": ("monetary_value", ["Low", "Medium", "High", "Premium"]),
    "tenure_segment": ("tenure_score", ["New", "Growing", "Established", "Loyal"]),
    "engagement_segment": (
        "engagement_score",
        ["Low", "Medium", "High", "Very High"],
    ),
}


def _safe_qcut(series: pd.Series, q: int, labels: list) -> pd.Series:
    """
    Legacy per-call qcut — kept ONLY as a fallback for ad-hoc exploration
    when no frozen segment_bins are supplied to create_engineered_features().
    Do not use this path for calibration/test/production scoring — it
    recomputes quantile edges on whatever slice of data it's given, which
    causes train/serve skew and can't even form quartiles on a single row.
    """
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


def _safe_qcut_edges(series: pd.Series, q: int, labels: list):
    """
    Fit quantile bin EDGES once — call this only on the training split.
    Returns (edges, labels) so the exact same boundaries can be reapplied
    to calibration/test/production data via apply_segment_bins(), instead
    of every caller recomputing its own quantiles.
    """
    try:
        _, edges = pd.qcut(series, q=q, labels=labels, duplicates="drop", retbins=True)
        n_bins = len(edges) - 1
        return edges, labels[:n_bins]
    except ValueError:
        pass
    for n_bins in range(min(q, series.nunique()), 1, -1):
        try:
            _, edges = pd.qcut(
                series,
                q=n_bins,
                labels=labels[:n_bins],
                duplicates="drop",
                retbins=True,
            )
            return edges, labels[:n_bins]
        except ValueError:
            continue
    return None, None  # degenerate column — caller falls back to "Mixed"


def fit_segment_bins(df: pd.DataFrame) -> dict:
    """
    Fit value/tenure/engagement segment bin edges ONCE, on the training
    split only. Save the returned dict (joblib.dump) alongside the model
    and preprocessor, and reuse it everywhere else — calibration set, test
    set, full-dataset scoring, and every prediction made by app.py — via
    apply_segment_bins() / create_engineered_features(df, segment_bins).
    Never refit these on calibration/test/production data.
    """
    bins = {}
    for seg_col, (src_col, labels) in SEGMENT_SPECS.items():
        edges, kept_labels = _safe_qcut_edges(df[src_col], q=4, labels=labels)
        bins[seg_col] = {"src_col": src_col, "edges": edges, "labels": kept_labels}
    return bins


def apply_segment_bins(df: pd.DataFrame, segment_bins: dict) -> pd.DataFrame:
    """
    Apply PRE-FITTED segment bin edges (from fit_segment_bins) to any
    slice of data — the full test set or a single customer row. Uses
    pd.cut with fixed edges instead of pd.qcut, so results are identical
    whether scoring a million rows or looking up one customer. Values
    outside the training range are clipped to the nearest edge instead
    of producing NaN.
    """
    df = df.copy()
    for seg_col, spec in segment_bins.items():
        edges, labels = spec["edges"], spec["labels"]
        src = df[spec["src_col"]]
        if edges is None:
            df[seg_col] = "Mixed"
            continue
        clipped = src.clip(lower=edges[0], upper=edges[-1])
        df[seg_col] = pd.cut(
            clipped, bins=edges, labels=labels, include_lowest=True, duplicates="drop"
        ).astype(str)
    return df


def create_engineered_features(df, segment_bins: dict | None = None):
    """
    Adds derived numeric features (plain arithmetic — always safe to
    recompute on any slice) plus the three segment columns.

    segment_bins: pass the dict returned by fit_segment_bins() (fit ONCE
    on the training split) to get consistent segment labels across
    train/calib/test/production. If omitted, falls back to the legacy
    per-call qcut for quick standalone exploration only — NEVER use the
    fallback for calibration, test scoring, or production inference, since
    it causes train/serve skew and can't form quartiles on a single row.
    """
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

    if segment_bins is not None:
        df = apply_segment_bins(df, segment_bins)
    else:
        for seg_col, (src_col, labels) in SEGMENT_SPECS.items():
            df[seg_col] = _safe_qcut(df[src_col], q=4, labels=labels)

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
    """
    NOTE: Always pass the RAW (uncalibrated) model here, never the
    CalibratedClassifierCV-wrapped one — TreeExplainer needs direct access
    to tree internals that the calibration wrapper hides. Isotonic
    calibration is monotonic, so SHAP direction/ranking computed on the raw
    model still reflects what drives the calibrated predictions.
    """
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

    df_raw = load_or_build_unified()

    X_raw = df_raw.drop(columns=[TARGET])
    y = df_raw[TARGET]

    # ── Split: train / calibration / test — on RAW data, BEFORE feature
    # engineering ──────────────────────────────────────────────────────
    # First peel off the test set (untouched, real-world distribution)
    X_train_full_raw, X_test_raw, y_train_full, y_test = train_test_split(
        X_raw,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    # Then peel off a calibration set from what remains (also real
    # distribution — SMOTE is applied only to what's left for training)
    X_train_raw, X_calib_raw, y_train, y_calib = train_test_split(
        X_train_full_raw,
        y_train_full,
        test_size=0.15,  # ~15% of train_full reserved for calibration
        random_state=config["data"]["random_state"],
        stratify=y_train_full,
    )
    print(
        f"\n[INFO] Train: {len(X_train_raw):,}  Calib: {len(X_calib_raw):,}  Test: {len(X_test_raw):,}"
    )

    # ── Fit segment bin edges ONCE, on the training split only, then
    # freeze them — calibration set, test set, single-customer lookups,
    # and any future scoring all reuse these exact same edges instead of
    # recomputing their own quantiles (see module docstring). ────────────
    segment_bins = fit_segment_bins(X_train_raw)

    X_train = create_engineered_features(X_train_raw, segment_bins)[MODEL_FEATURES]
    X_calib = create_engineered_features(X_calib_raw, segment_bins)[MODEL_FEATURES]
    X_test = create_engineered_features(X_test_raw, segment_bins)[MODEL_FEATURES]

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_calib_proc = preprocessor.transform(X_calib)  # real distribution, no SMOTE
    X_test_proc = preprocessor.transform(X_test)  # real distribution, no SMOTE

    ohe_names = (
        preprocessor.named_transformers_["cat"]["encoder"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    feature_names = num_cols + ohe_names

    # SMOTE only touches the training data — calib/test stay at real ~21% churn
    X_train_bal, y_train_bal = handle_imbalance(X_train_proc, y_train.values)

    results, best_model, best_name = train_and_evaluate(
        X_train_bal, X_test_proc, y_train_bal, y_test.values
    )

    # ── Calibrate the winning model on the untouched calibration set ────
    # Manual isotonic calibration via CalibratedModel (src/calibration.py)
    # — avoids CalibratedClassifierCV/FrozenEstimator, which embed sklearn-
    # version-specific internals into the pickle and break when the model
    # is loaded in a different Python environment than it was trained in.
    # CalibratedModel lives in its own module (never run as __main__) so it
    # always pickles with a stable, consistent import path.
    print(
        f"\n[INFO] Calibrating {best_name} on held-out (non-SMOTE) calibration set..."
    )
    raw_calib_probs = best_model.predict_proba(X_calib_proc)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib_probs, y_calib.values)
    calibrated_model = CalibratedModel(best_model, calibrator)

    # Sanity check: compare avg predicted prob vs actual churn rate on test set
    raw_probs = best_model.predict_proba(X_test_proc)[:, 1]
    calib_probs = calibrated_model.predict_proba(X_test_proc)[:, 1]
    actual_rate = y_test.mean()
    print(f"[INFO] Actual test churn rate     : {actual_rate:.4f}")
    print(f"[INFO] Raw (uncalibrated) avg prob: {raw_probs.mean():.4f}")
    print(f"[INFO] Calibrated avg prob        : {calib_probs.mean():.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        calibrated_model, MODEL_DIR / "unified_best_model.pkl"
    )  # calibrated, for predictions
    joblib.dump(
        best_model, MODEL_DIR / "unified_raw_model.pkl"
    )  # uncalibrated, for SHAP
    joblib.dump(preprocessor, MODEL_DIR / "unified_preprocessor.pkl")
    joblib.dump(
        segment_bins, MODEL_DIR / "unified_segment_bins.pkl"
    )  # frozen bin edges
    print(f"[INFO] Calibrated model saved -> {MODEL_DIR / 'unified_best_model.pkl'}")
    print(f"[INFO] Raw model saved (for SHAP) -> {MODEL_DIR / 'unified_raw_model.pkl'}")
    print(f"[INFO] Segment bin edges saved -> {MODEL_DIR / 'unified_segment_bins.pkl'}")

    # Score full dataset using the calibrated model — engineer features with
    # the SAME frozen bins fit on the training split (not a fresh qcut).
    df_for_scoring = create_engineered_features(df_raw, segment_bins)
    X_all_proc = preprocessor.transform(df_for_scoring[MODEL_FEATURES])
    scored_df = score_and_segment(calibrated_model, X_all_proc, df_raw)

    shap_sample_size = min(config["shap"]["sample_size"], X_test_proc.shape[0])
    # Explain using the RAW model — TreeExplainer can't see through the
    # CalibratedClassifierCV wrapper. See explain_with_shap() docstring.
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
        "best_model": calibrated_model,
        "raw_model": best_model,
        "preprocessor": preprocessor,
        "segment_bins": segment_bins,
        "results": results,
        "shap_df": shap_df,
    }


if __name__ == "__main__":
    run_unified_pipeline()

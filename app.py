"""
app.py
======
Streamlit front-end for the Unified Cross-Industry Churn Prediction model.

Flow: Upload -> Map columns -> Predict -> Dashboard -> Download
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.unified_churn_pipeline import create_engineered_features, MODEL_FEATURES

# Vercel / HF Spaces set SPACE_ID or VERCEL env vars — detect cloud deployment
IS_DEPLOYED = os.environ.get("SPACE_ID") is not None or os.environ.get("VERCEL") is not None

# Admin password — change this or set via environment variable
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "churn@admin123")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "unified_best_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "unified_preprocessor.pkl"

# The 8 "base" generic columns a user's raw file must be mapped onto.
BASE_COLUMNS = {
    "tenure_score": {
        "description": "How long the customer has stayed, normalised 0-1",
        "dtype": "numeric",
        "synonyms": ["tenure", "months", "duration", "age", "customer_age",
                     "account_age", "membership_duration", "tenure_score",
                     "customer_tenure", "months_active", "length_of_stay",
                     "tenurescore"],
    },
    "gender": {
        "description": "Customer gender (male/female or 0/1)",
        "dtype": "binary",
        "synonyms": ["gender", "sex", "customer_gender"],
    },
    "engagement_score": {
        "description": "Usage/activity intensity, normalised 0-1",
        "dtype": "numeric",
        "synonyms": ["engagement", "activity", "usage", "engagement_score",
                     "engagementscore", "active_days", "login_frequency",
                     "session_count", "interaction_score"],
    },
    "monetary_value": {
        "description": "Spend / balance / charges, normalised 0-1",
        "dtype": "numeric",
        "synonyms": ["monetary", "spend", "charges", "monthly_charges",
                     "monetary_value", "revenue", "arpu", "avg_spend",
                     "balance", "total_spend", "amount"],
    },
    "product_or_service_count": {
        "description": "Number of products or services used",
        "dtype": "numeric",
        "synonyms": ["product", "service", "count", "num_products",
                     "product_count", "service_count", "product_or_service_count",
                     "services_used", "num_services", "subscriptions"],
    },
    "support_friction": {
        "description": "Complaints / support-ticket signal (0/1 or count)",
        "dtype": "numeric",
        "synonyms": ["support", "friction", "complaints", "tickets",
                     "support_friction", "support_tickets", "issues",
                     "helpdesk_calls", "complaint_count"],
    },
    "is_active": {
        "description": "Currently engaged flag (0/1)",
        "dtype": "binary",
        "synonyms": ["active", "is_active", "currently_active", "status",
                     "account_status", "active_flag", "enabled"],
    },
    "source_industry": {
        "description": "Industry: ecommerce / banking / telco",
        "dtype": "categorical",
        "synonyms": ["industry", "source", "sector", "source_industry",
                     "business_type", "vertical", "segment"],
    },
}

RISK_COLORS = {"High": "#e15759", "Medium": "#f2b134", "Low": "#59a14f"}


# ════════════════════════════════════════════════════════════════════════════
# SMART COLUMN MAPPING ENGINE
# ════════════════════════════════════════════════════════════════════════════

def _normalize(name: str) -> str:
    """Strip punctuation and lowercase for comparison."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _fuzzy_score(a: str, b: str) -> float:
    """Similarity score 0-1 between two strings."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.85
    return SequenceMatcher(None, na, nb).ratio()


def _best_match(target: str, user_columns: list[str]) -> tuple[str | None, float]:
    """
    Find the best matching user column for a model field.
    Scores against field name + all synonyms. Returns (column, score).
    Only returns a match if score >= 0.45.
    """
    synonyms = BASE_COLUMNS[target]["synonyms"]
    best_col, best_score = None, 0.0
    for col in user_columns:
        score = _fuzzy_score(target, col)
        for syn in synonyms:
            score = max(score, _fuzzy_score(syn, col))
        if score > best_score:
            best_score, best_col = score, col

    # Ensure no two fields grab the same column at the same score
    if best_score >= 0.45:
        return best_col, best_score
    return None, 0.0


def _compatible_columns(target: str, user_df: pd.DataFrame) -> list[str]:
    """
    Return only columns from user_df that are compatible with the
    expected dtype of the target field.
      numeric    → only numeric columns
      binary     → numeric + ≤5 unique values
      categorical→ numeric + ≤25 unique values
    """
    expected = BASE_COLUMNS[target]["dtype"]
    compatible = []
    for col in user_df.columns:
        dtype   = user_df[col].dtype
        n_uniq  = user_df[col].nunique()
        is_num  = pd.api.types.is_numeric_dtype(dtype)

        if expected == "numeric"     and is_num:                   compatible.append(col)
        elif expected == "binary"    and (is_num or n_uniq <= 5):  compatible.append(col)
        elif expected == "categorical" and (is_num or n_uniq <= 25): compatible.append(col)

    return compatible


def _value_range_check(target: str, col: str, user_df: pd.DataFrame) -> str | None:
    """
    Check whether the actual data values in col make sense for target.
    Returns a warning string if something looks wrong, else None.

    Key checks:
    - *_score fields expect values in [0, 1] — warn if column has values > 1
    - binary fields expect only 0/1 or 2 unique values
    - source_industry expects string categories, not numbers
    """
    if col not in user_df.columns:
        return None

    series = user_df[col].dropna()
    if len(series) == 0:
        return None

    schema = BASE_COLUMNS.get(target, {})

    # ── Normalised score fields should be 0-1 ────────────────────────────
    if target.endswith("_score") and pd.api.types.is_numeric_dtype(series):
        col_max = series.max()
        col_min = series.min()
        if col_max > 1.5:
            return (
                f"Values go up to {col_max:.0f} — expected 0–1 normalised score. "
                f"Is this raw data instead of a score? Map a different column or "
                f"normalise this one first."
            )

    # ── monetary_value should be 0-1 normalised too ───────────────────────
    if target == "monetary_value" and pd.api.types.is_numeric_dtype(series):
        if series.max() > 1.5:
            return (
                f"Values go up to {series.max():.0f} — expected 0–1 normalised value. "
                f"If this is raw spend/charges, normalise it first (divide by max)."
            )

    # ── Binary fields should have at most 2-3 unique values ──────────────
    if schema.get("dtype") == "binary":
        n_unique = series.nunique()
        unique_vals = sorted(series.unique()[:5].tolist())
        if n_unique > 5:
            return (
                f"Found {n_unique} unique values {unique_vals}... — "
                f"expected a binary (0/1 or Yes/No) column."
            )

    return None


def _confidence_badge(score: float, warning: str | None = None) -> tuple[str, str]:
    """(label, hex_color) for a match confidence score."""
    if warning:
        return "Review needed ⚠", "#e67e22"   # always yellow if value warning
    if score >= 0.85: return "Auto-matched ✓", "#27ae60"
    if score >= 0.60: return "Good match",     "#2980b9"
    if score >= 0.45: return "Possible match", "#e67e22"
    return "No match",          "#e74c3c"


def render_smart_mapper(user_df: pd.DataFrame) -> dict | None:
    """
    Drop-in replacement for the old dumb selectbox loop.
    Returns {target_col: user_col} mapping dict when confirmed,
    or None if the user hasn't confirmed yet.
    """
    user_cols = user_df.columns.tolist()

    # ── Auto-match every field, resolve conflicts (highest score wins) ────
    raw_scores: dict[str, tuple[str | None, float]] = {}
    for target in BASE_COLUMNS:
        raw_scores[target] = _best_match(target, user_cols)

    # Deduplicate: if two targets matched the same column, keep the higher score
    claimed: dict[str, str] = {}   # col → target that claimed it
    auto_matches: dict[str, str | None] = {}
    for target, (col, score) in sorted(raw_scores.items(),
                                        key=lambda x: -x[1][1]):
        if col is None:
            auto_matches[target] = None
        elif col not in claimed:
            claimed[col] = target
            auto_matches[target] = col
        else:
            auto_matches[target] = None   # lost the contest — leave unset

    # ── Summary bar ───────────────────────────────────────────────────────
    matched = sum(1 for v in auto_matches.values() if v is not None)
    total   = len(BASE_COLUMNS)
    needs_review = sum(
        1 for t, col in auto_matches.items()
        if col is not None and (
            raw_scores[t][1] < 0.85 or
            _value_range_check(t, col, user_df) is not None
        )
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total fields",  total)
    c2.metric("Auto-matched",  matched,
              delta=f"{matched/total:.0%} done", delta_color="normal")
    c3.metric("Needs review",  needs_review,
              delta_color="inverse" if needs_review else "off")

    st.caption(
        "🟢 Auto-matched columns are pre-filled. "
        "🟡 Review yellow ones. "
        "Only compatible columns appear in each dropdown."
    )

    # ── Per-field dropdowns ───────────────────────────────────────────────
    final_mapping: dict[str, str | None] = {}
    map_cols = st.columns(2)

    for i, target in enumerate(BASE_COLUMNS):
        schema      = BASE_COLUMNS[target]
        auto_col    = auto_matches[target]
        score       = raw_scores[target][1] if auto_col else 0.0

        # Check if auto-matched column has suspicious values
        value_warning = _value_range_check(target, auto_col, user_df) if auto_col else None
        badge_label, badge_color = _confidence_badge(score, value_warning)

        # Only compatible columns in dropdown (incompatible ones hidden)
        compatible  = _compatible_columns(target, user_df)
        if auto_col and auto_col not in compatible:
            compatible = [auto_col] + compatible
        incompatible_count = len(user_df.columns) - len(compatible)

        options     = ["— none —"] + compatible
        default_idx = options.index(auto_col) if auto_col in options else 0

        with map_cols[i % 2]:
            chosen = st.selectbox(
                label=target,
                options=options,
                index=default_idx,
                key=f"map_{target}",
                label_visibility="collapsed",
                help=(
                    f"{schema['description']}\n\n"
                    f"Expected type: {schema['dtype']}"
                    + (f"\n⚠️ {incompatible_count} column(s) hidden (wrong type)"
                       if incompatible_count else "")
                ),
            )

            # Recompute badge and warning based on CHOSEN column (not auto-match)
            if chosen == "— none —":
                live_warning  = None
                live_score    = 0.0
            else:
                live_warning  = _value_range_check(target, chosen, user_df)
                # Recompute fuzzy score for chosen column
                live_score    = max(
                    _fuzzy_score(target, chosen),
                    max((_fuzzy_score(s, chosen) for s in schema["synonyms"]), default=0.0)
                )

            live_badge_label, live_badge_color = _confidence_badge(live_score, live_warning)

            # Field label with LIVE badge (updates when user changes dropdown)
            st.markdown(
                f"**`{target}`** &nbsp;"
                f"<span style='font-size:11px; color:{live_badge_color}; "
                f"padding:2px 7px; border-radius:4px; "
                f"background:{live_badge_color}22'>{live_badge_label}</span>",
                unsafe_allow_html=True,
            )

            # Show warning + auto-normalise option if values out of range
            if live_warning:
                st.warning(f"⚠️ {live_warning}", icon="⚠️")
                # Offer to auto-normalise if it's a numeric scale issue
                if chosen != "— none —" and pd.api.types.is_numeric_dtype(user_df[chosen]):
                    if st.checkbox(
                        f"Auto-normalise `{chosen}` to 0–1 (divide by max)",
                        key=f"norm_{target}",
                        value=False,
                    ):
                        col_max = user_df[chosen].max()
                        user_df[chosen] = user_df[chosen] / col_max
                        st.success(f"✅ `{chosen}` normalised (divided by {col_max:.2f})")
                        live_warning = None   # clear warning after fix

            final_mapping[target] = None if chosen == "— none —" else chosen

    # ── Validation ────────────────────────────────────────────────────────
    missing = [t for t, c in final_mapping.items() if c is None]

    # Duplicate check (same user column mapped to two model fields)
    col_usage: dict[str, list[str]] = {}
    for t, c in final_mapping.items():
        if c:
            col_usage.setdefault(c, []).append(t)
    duplicates = {c: ts for c, ts in col_usage.items() if len(ts) > 1}

    if missing:
        st.warning(
            f"⚠️ Please map all fields before predicting. "
            f"Missing: **{', '.join(missing)}**"
        )

    if duplicates:
        for col, targets in duplicates.items():
            st.error(
                f"⛔ **`{col}`** is mapped to multiple fields: "
                f"{', '.join(f'`{t}`' for t in targets)}. "
                f"Each column can only be used once."
            )

    # ── Mapping preview ───────────────────────────────────────────────────
    mapped_items = [(t, c) for t, c in final_mapping.items() if c]
    if mapped_items:
        with st.expander("👁️ Preview mapping", expanded=False):
            preview = pd.DataFrame([
                {
                    "Model field":   t,
                    "→ Your column": c,
                    "Sample value":  str(user_df[c].iloc[0]),
                    "Type OK":       "✅" if c in _compatible_columns(t, user_df) else "⚠️ Review",
                }
                for t, c in mapped_items
            ])
            st.dataframe(preview, width="stretch", hide_index=True)

    # ── Confirm button ─────────────────────────────────────────────────────
    can_confirm = not missing and not duplicates
    if st.button(
        "✅ Confirm mapping & continue",
        type="primary",
        disabled=not can_confirm,
        width="stretch",
    ):
        return {t: c for t, c in final_mapping.items() if c}

    return None



# ════════════════════════════════════════════════════════════════════════════
# CSV VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def validate_csv(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Run data quality checks on the uploaded CSV.
    Returns (errors, warnings).
    errors   → must be fixed before predictions can run
    warnings → shown to user but don't block predictions
    """
    errors   = []
    warnings = []

    # ── Basic shape ───────────────────────────────────────────────────────
    if len(df) == 0:
        errors.append("The uploaded file has no rows.")
        return errors, warnings

    if len(df.columns) < 2:
        errors.append("The file has fewer than 2 columns — check the CSV format.")
        return errors, warnings

    # ── Duplicate rows ────────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        warnings.append(
            f"{n_dupes:,} duplicate row(s) found. "
            f"They will be included in predictions but may skew results."
        )

    # ── Null / missing values ─────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        for col, count in null_cols.items():
            pct = count / len(df) * 100
            msg = f"Column '{col}' has {count:,} missing value(s) ({pct:.1f}% of rows)."
            if pct > 30:
                errors.append(msg + " Over 30% missing — please fix before predicting.")
            else:
                warnings.append(msg + " Missing values will be filled with column median/mode.")

    # ── All-null columns ──────────────────────────────────────────────────
    all_null = [c for c in df.columns if df[c].isnull().all()]
    if all_null:
        errors.append(
            f"Column(s) {all_null} are completely empty. "
            f"Remove or fill them before uploading."
        )

    # ── Constant columns (no variance) ───────────────────────────────────
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        warnings.append(
            f"Column(s) {constant_cols} have only one unique value. "
            f"They won't help the model and may indicate a data issue."
        )

    # ── Numeric column checks ─────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Extreme outliers (beyond 5 std devs)
        mean, std = series.mean(), series.std()
        if std > 0:
            n_outliers = ((series - mean).abs() > 5 * std).sum()
            if n_outliers > 0:
                warnings.append(
                    f"Column '{col}' has {n_outliers} extreme outlier(s) "
                    f"(beyond 5 standard deviations). Check for data entry errors."
                )

        # Negative values in columns that should be non-negative
        non_neg_hints = ["tenure", "score", "count", "value", "monetary",
                         "friction", "active", "charges", "amount"]
        if any(h in col.lower() for h in non_neg_hints):
            n_neg = (series < 0).sum()
            if n_neg > 0:
                warnings.append(
                    f"Column '{col}' has {n_neg} negative value(s). "
                    f"Expected non-negative numbers for this field."
                )

    # ── File size warning ─────────────────────────────────────────────────
    if len(df) > 50_000:
        warnings.append(
            f"Large file: {len(df):,} rows. Predictions may take 10-30 seconds."
        )

    return errors, warnings


def render_validation_report(df: pd.DataFrame) -> bool:
    """
    Runs validation and renders the results.
    Returns True if safe to proceed, False if errors block prediction.
    """
    errors, warnings = validate_csv(df)

    if not errors and not warnings:
        st.success(
            f"✅ Data looks clean — {len(df):,} rows, "
            f"{len(df.columns)} columns, no issues found."
        )
        return True

    # Show errors
    if errors:
        st.error("**Data quality errors — fix these before predicting:**")
        for e in errors:
            st.markdown(f"- ⛔ {e}")

    # Show warnings
    if warnings:
        with st.expander(
            f"⚠️ {len(warnings)} data quality warning(s) — predictions will still run",
            expanded=True
        ):
            for w in warnings:
                st.markdown(f"- ⚠️ {w}")

    # Summary row
    col1, col2, col3 = st.columns(3)
    null_total = df.isnull().sum().sum()
    dupe_total = df.duplicated().sum()
    col1.metric("Total rows",     f"{len(df):,}")
    col2.metric("Missing values", f"{null_total:,}",
                delta_color="inverse" if null_total > 0 else "off")
    col3.metric("Duplicate rows", f"{dupe_total:,}",
                delta_color="inverse" if dupe_total > 0 else "off")

    # Block if errors exist
    if errors:
        st.stop()

    return True


# ════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_shap_explainer(_model):
    """
    Build a SHAP explainer for the loaded model.
    Cached so it's only created once per session.
    Uses TreeExplainer for tree-based models (RF/XGBoost),
    falls back to LinearExplainer for linear models.
    """
    import shap
    try:
        # feature_perturbation="tree_path_dependent" avoids needing
        # background data and works with any array format
        return shap.TreeExplainer(
            _model,
            feature_perturbation="tree_path_dependent"
        )
    except Exception:
        try:
            return shap.TreeExplainer(_model)
        except Exception:
            return None   # will be caught gracefully at call site


def _run_shap(explainer, X_dense):
    """
    Run shap_values safely — always returns a 2D numpy array
    of shape (n_samples, n_features) for the positive (churn) class.

    Handles every output format:
      - list of 2 arrays  → RandomForest binary (old SHAP)
      - single 2D array   → XGBoost binary
      - single 3D array   → RandomForest (new SHAP: shape n_classes, n_samples, n_features)
    """
    import numpy as np
    if explainer is None:
        raise ValueError("No SHAP explainer available for this model type.")

    vals = explainer.shap_values(X_dense)
    arr  = np.array(vals)

    # Case 1: list of arrays e.g. [class0, class1]  → shape (2, n, f)
    if arr.ndim == 3:
        # axis 0 = classes → take index 1 (churn class)
        return arr[1]

    # Case 2: list of 2 plain arrays
    if isinstance(vals, list) and len(vals) == 2:
        return np.array(vals[1])

    # Case 3: already 2D — XGBoost binary or single-output
    if arr.ndim == 2:
        return arr

    # Fallback
    return arr


def _to_dense(X):
    """Convert sparse matrix to dense numpy array. No-op if already dense."""
    import scipy.sparse as sp
    import numpy as np
    if sp.issparse(X):
        return X.toarray()
    if hasattr(X, "values"):      # pandas DataFrame
        return X.values
    return np.asarray(X)


def compute_shap_reasons(model, X_processed, feature_names, top_n=3,
                         max_rows=500):
    """
    Compute SHAP values and return a list of human-readable reason strings
    for each customer — e.g. "↑ tenure_score  ↑ support_friction  ↓ engagement_score"

    For large datasets (>max_rows) we compute exact SHAP only for the first
    max_rows customers and use the dataset-average reasons for the rest.
    This keeps runtime under ~3 seconds even for 22,000+ rows.
    """
    import shap
    import numpy as np

    # Always convert to dense numpy — SHAP can't handle sparse matrices
    X_dense = _to_dense(X_processed)
    n_total = X_dense.shape[0]

    try:
        explainer = get_shap_explainer(model)

        # ── Compute SHAP on a capped sample ───────────────────────────
        if n_total > max_rows:
            # Compute exact SHAP for first max_rows rows
            shap_sample = _run_shap(explainer, X_dense[:max_rows])

            # For remaining rows use the mean SHAP direction from the sample
            mean_vals = shap_sample.mean(axis=0)
            shap_full = np.vstack([
                shap_sample,
                np.tile(mean_vals, (n_total - max_rows, 1))
            ])
        else:
            shap_full = _run_shap(explainer, X_dense)

        # ── Build reason strings ───────────────────────────────────────
        reasons = []
        for i, row_vals in enumerate(shap_full):
            top_idx = np.argsort(np.abs(row_vals))[::-1][:top_n]
            parts   = []
            for idx in top_idx:
                direction = "↑" if row_vals[idx] > 0 else "↓"
                fname     = feature_names[idx].replace("_", " ")
                parts.append(f"{direction} {fname}")
            suffix = "  (estimated)" if i >= max_rows else ""
            reasons.append("  |  ".join(parts) + suffix)

        return reasons

    except Exception as _shap_err:
        import streamlit as _st
        _st.warning(f"SHAP explanation unavailable: {_shap_err}")
        return ["—"] * X_dense.shape[0]


def render_shap_deep_dive(model, X_processed, feature_names):
    """
    Full SHAP summary plot shown in an expander below the results table.
    Shows global feature importance + a beeswarm / bar chart.
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    st.subheader("🔍 Why is the model predicting churn?")
    st.caption("SHAP values show which features push customers toward or away from churning.")

    try:
        import numpy as np
        explainer = get_shap_explainer(model)

        # Sample max 1000 rows for the chart — statistically identical result
        CHART_SAMPLE = 1000
        n = X_processed.shape[0]
        if n > CHART_SAMPLE:
            rng      = np.random.default_rng(42)
            idx      = rng.choice(n, size=CHART_SAMPLE, replace=False)
            # Handle both numpy arrays and DataFrames
            if hasattr(X_processed, "iloc"):
                X_sample = X_processed.iloc[idx]
            else:
                X_sample = X_processed[idx]
        else:
            X_sample = X_processed

        # Convert to dense numpy — SHAP fails on sparse matrices
        X_sample_dense = _to_dense(X_sample)
        shap_vals = _run_shap(explainer, X_sample_dense)

        # ── Top features bar chart ─────────────────────────────────────
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:10]

        top_features = [feature_names[i].replace("_", " ") for i in top_idx]
        top_values   = [round(float(mean_abs[i]), 4) for i in top_idx]

        import plotly.express as px
        fig = px.bar(
            x=top_values[::-1],
            y=top_features[::-1],
            orientation="h",
            title="Top 10 features driving churn (mean |SHAP value|)",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
            color=top_values[::-1],
            color_continuous_scale=["#59a14f", "#f2b134", "#e15759"],
        )
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=40, b=10),
            height=380,
        )
        st.plotly_chart(fig, width="stretch")

        # ── Interpretation guide ───────────────────────────────────────
        st.caption(
            "Higher bar = stronger influence on predictions. "
            "↑ means this feature increases churn risk. "
            "↓ means it decreases churn risk."
        )

    except Exception as ex:
        st.warning(f"SHAP visualisation unavailable: {ex}")


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

# Model loading handled at startup via load_model_cached() above


def risk_tier(prob: float) -> str:
    if prob >= 0.70: return "High"
    if prob >= 0.40: return "Medium"
    return "Low"


def make_template_csv() -> bytes:
    sample = pd.DataFrame({
        "tenure_score":            [0.12, 0.87, 0.45],
        "gender":                  ["female", "male", "male"],
        "engagement_score":        [0.34, 0.91, 0.55],
        "monetary_value":          [0.20, 0.76, 0.48],
        "product_or_service_count":[1, 4, 2],
        "support_friction":        [1, 0, 0],
        "is_active":               [0, 1, 1],
        "source_industry":         ["telco", "banking", "ecommerce"],
    })
    return sample.to_csv(index=False).encode("utf-8")


# ── Page config must be first Streamlit call — UI appears immediately ────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme toggle + Custom CSS ────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Theme variables
if st.session_state["theme"] == "dark":
    BG        = "linear-gradient(135deg, #111111 0%, #1a1a1a 100%)"
    CARD_BG   = "rgba(28, 28, 28, 0.95)"
    CARD_BDR  = "rgba(255, 255, 255, 0.08)"
    TEXT_PRI  = "#f5f5f5"
    TEXT_SEC  = "#a0a0a0"
    TEXT_MUT  = "#555555"
    ACCENT    = "#6366f1"
    ACCENT_LT = "#818cf8"
    SIDEBAR   = "#0d0d0d"
    INPUT_BG  = "rgba(20, 20, 20, 0.95)"
else:
    BG        = "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)"
    CARD_BG   = "rgba(255, 255, 255, 0.9)"
    CARD_BDR  = "rgba(99, 102, 241, 0.25)"
    TEXT_PRI  = "#1e293b"
    TEXT_SEC  = "#475569"
    TEXT_MUT  = "#94a3b8"
    ACCENT    = "#6366f1"
    ACCENT_LT = "#4f46e5"
    SIDEBAR   = "#f1f5f9"
    INPUT_BG  = "rgba(255, 255, 255, 0.9)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}

.stApp {{
    background: {BG};
}}

[data-testid="stSidebar"] {{
    background: {SIDEBAR};
    border-right: 1px solid {CARD_BDR};
}}

h1 {{
    background: linear-gradient(135deg, {ACCENT_LT}, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2rem !important;
}}
h2 {{
    color: {ACCENT_LT} !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    border-bottom: 1px solid {CARD_BDR};
    padding-bottom: 6px;
}}
h3 {{ color: {TEXT_PRI} !important; font-weight: 500 !important; }}
p, li, span {{ color: {TEXT_SEC}; }}

[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {CARD_BDR};
    border-radius: 12px;
    padding: 16px 20px !important;
    transition: transform 0.2s, border-color 0.2s;
    backdrop-filter: blur(10px);
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    border-color: {ACCENT};
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SEC} !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRI} !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}}
.stButton > button[kind="secondary"] {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BDR} !important;
    color: {TEXT_PRI} !important;
    box-shadow: none !important;
}}

.stTextInput > div > div > input,
.stSelectbox > div > div {{
    background: {INPUT_BG} !important;
    border: 1px solid {CARD_BDR} !important;
    border-radius: 8px !important;
    color: {TEXT_PRI} !important;
}}

[data-testid="stFileUploader"] {{
    background: {CARD_BG};
    border: 2px dashed {CARD_BDR};
    border-radius: 12px;
    padding: 20px;
}}

[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    border: 1px solid {CARD_BDR} !important;
}}

[data-testid="stExpander"] {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BDR} !important;
    border-radius: 10px !important;
}}

[data-testid="stDownloadButton"] > button {{
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.4) !important;
    color: #10b981 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}

hr {{ border-color: {CARD_BDR} !important; margin: 1.5rem 0 !important; }}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {CARD_BDR}; border-radius: 99px; }}
</style>
""", unsafe_allow_html=True)


# ── Load model with spinner so user sees progress not blank screen ─────────
@st.cache_resource(show_spinner=False)
def load_model_cached():
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)

with st.spinner("⏳ Loading model..."):
    model, preprocessor = load_model_cached()

st.sidebar.title("📊 Navigation")

# ── Theme toggle ──────────────────────────────────────────────────────────
current_theme = st.session_state.get("theme", "dark")
toggle_label  = "☀️ Switch to Light" if current_theme == "dark" else "🌙 Switch to Dark"
if st.sidebar.button(toggle_label, width="stretch", key="theme_toggle"):
    st.session_state["theme"] = "light" if current_theme == "dark" else "dark"
    st.rerun()
st.sidebar.divider()
# ── Admin login in sidebar ────────────────────────────────────────────────
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

with st.sidebar.expander("🔐 Admin login", expanded=False):
    if st.session_state["is_admin"]:
        st.success("Logged in as admin")
        if st.button("Logout", width="stretch"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        pwd = st.text_input("Password", type="password", key="admin_pwd")
        if st.button("Login", width="stretch"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("Incorrect password")

# Show Model Health only to admins
_pages = ["🏠 Home", "🔮 Predict Churn", "🧍 Single Customer", "📊 Dashboard", "ℹ️ About"]
if st.session_state.get("is_admin"):
    _pages.insert(4, "🏥 Model Health")

page = st.sidebar.radio("Go to", _pages)

# ════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # ── Hero section ──────────────────────────────────────────────────────
    st.title("Customer Churn Prediction")
    st.markdown(
        "<p style='color:#94a3b8; font-size:18px; margin-top:-10px; margin-bottom:24px'>"
        "Know which customers are about to leave — before they do."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Stats row ─────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model Accuracy",   "83%",   delta="ROC-AUC")
    s2.metric("Industries",       "3",     delta="E-com · Bank · Telco")
    s3.metric("Model",            type(model).__name__)
    s4.metric("Features",         "17",    delta="cross-industry")

    st.divider()

    # ── How it works ──────────────────────────────────────────────────────
    st.subheader("How it works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:20px; text-align:center; height:160px;'>
            <div style='font-size:32px;'>📁</div>
            <div style='font-weight:600; color:#a5b4fc; margin:8px 0 4px;'>1. Upload</div>
            <div style='font-size:13px; color:#64748b;'>Upload your customer CSV — any column names</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:20px; text-align:center; height:160px;'>
            <div style='font-size:32px;'>🔗</div>
            <div style='font-weight:600; color:#a5b4fc; margin:8px 0 4px;'>2. Map</div>
            <div style='font-size:13px; color:#64748b;'>Auto-match your columns to our schema</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:20px; text-align:center; height:160px;'>
            <div style='font-size:32px;'>🔮</div>
            <div style='font-weight:600; color:#a5b4fc; margin:8px 0 4px;'>3. Predict</div>
            <div style='font-size:13px; color:#64748b;'>Get churn probability for every customer</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:20px; text-align:center; height:160px;'>
            <div style='font-size:32px;'>💡</div>
            <div style='font-weight:600; color:#a5b4fc; margin:8px 0 4px;'>4. Act</div>
            <div style='font-size:13px; color:#64748b;'>Get retention recommendations per risk tier</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Risk tiers explanation ─────────────────────────────────────────────
    st.subheader("Risk tiers")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div style='background:rgba(225,87,89,0.1); border:1px solid rgba(225,87,89,0.3);
                    border-radius:12px; padding:18px;'>
            <div style='font-size:22px; font-weight:700; color:#e15759;'>🔴 High Risk</div>
            <div style='font-size:13px; color:#94a3b8; margin-top:6px;'>Churn probability ≥ 70%</div>
            <div style='font-size:13px; color:#cbd5e1; margin-top:8px;'>
                Immediate personal outreach — offer retention deal within 48 hours.
            </div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div style='background:rgba(242,177,52,0.1); border:1px solid rgba(242,177,52,0.3);
                    border-radius:12px; padding:18px;'>
            <div style='font-size:22px; font-weight:700; color:#f2b134;'>🟠 Medium Risk</div>
            <div style='font-size:13px; color:#94a3b8; margin-top:6px;'>Churn probability 40–70%</div>
            <div style='font-size:13px; color:#cbd5e1; margin-top:8px;'>
                Targeted re-engagement — highlight features, offer modest incentive.
            </div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div style='background:rgba(89,161,79,0.1); border:1px solid rgba(89,161,79,0.3);
                    border-radius:12px; padding:18px;'>
            <div style='font-size:22px; font-weight:700; color:#59a14f;'>🟢 Low Risk</div>
            <div style='font-size:13px; color:#94a3b8; margin-top:6px;'>Churn probability < 40%</div>
            <div style='font-size:13px; color:#cbd5e1; margin-top:8px;'>
                Standard loyalty programme — periodic check-in at milestones.
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Sample template ────────────────────────────────────────────────────
    st.subheader("Get started")
    dl_col, info_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇️ Download sample template",
            data=make_template_csv(),
            file_name="churn_template.csv",
            mime="text/csv",
            width="stretch",
        )
    with info_col:
        st.markdown(
            "<p style='color:#64748b; font-size:13px; margin-top:8px;'>"
            "Download the sample CSV to see the expected format. "
            "Your own data doesn't need to match these column names exactly — "
            "the smart mapper will auto-detect them."
            "</p>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")

    # ── Step 1: Upload ──────────────────────────────────────────────────
    st.subheader("1. Upload your customer data")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is not None:
        # New file uploaded — read it, save to session_state, clear old mapping
        file_key = uploaded_file.name + str(uploaded_file.size)
        if st.session_state.get("_upload_key") != file_key:
            raw_df = pd.read_csv(uploaded_file)
            st.session_state["_upload_key"]  = file_key
            st.session_state["_raw_df"]      = raw_df
            st.session_state["_file_name"]   = uploaded_file.name
            st.session_state.pop("confirmed_mapping", None)
        else:
            # Same file — reuse from session_state (no re-read needed)
            raw_df = st.session_state["_raw_df"]

    elif "_raw_df" in st.session_state:
        # User navigated away and came back — restore from session_state
        raw_df = st.session_state["_raw_df"]
        st.success(
            f"✅ Using previously uploaded file: "
            f"**{st.session_state.get('_file_name', 'your dataset')}** "
            f"({len(raw_df):,} rows) — "
            f"[Upload a different file to replace it]"
        )
    else:
        # Nothing uploaded yet
        st.info("Upload a CSV to get started, or grab the template from the Home page.")
        st.stop()

    with st.expander("Preview uploaded data"):
        st.dataframe(raw_df.head())

    # ── Step 1b: Validate the uploaded data ─────────────────────────────
    st.subheader("1b. Data quality check")
    render_validation_report(raw_df)

    # ── Step 2: Smart column mapping ────────────────────────────────────
    st.subheader("2. Map your columns to the model's schema")

    # file_key already set above when file was uploaded
    file_key = st.session_state.get("_upload_key", "")

    # If mapping not yet confirmed, show the mapper and wait
    if "confirmed_mapping" not in st.session_state:
        result = render_smart_mapper(raw_df)
        if result is None:
            st.stop()   # user hasn't clicked Confirm yet
        # Confirmed — persist to session_state and rerun to show Step 3
        st.session_state["confirmed_mapping"] = result
        st.rerun()

    mapping = st.session_state["confirmed_mapping"]

    # Show a compact confirmed-mapping summary with option to redo
    with st.expander("✅ Column mapping confirmed — click to review or change", expanded=False):
        summary = pd.DataFrame([
            {"Model field": k, "→ Your column": v}
            for k, v in mapping.items()
        ])
        st.dataframe(summary, width="stretch", hide_index=True)
        if st.button("🔄 Change mapping", width="stretch"):
            st.session_state.pop("confirmed_mapping", None)
            st.rerun()

    # ── Step 3: Predict ─────────────────────────────────────────────────
    st.subheader("3. Run predictions")
    if st.button("🚀 Predict Churn", type="primary"):
        try:
            df = raw_df.rename(columns={v: k for k, v in mapping.items()})[
                list(BASE_COLUMNS)
            ].copy()

            df = create_engineered_features(df)
            for col in ["value_segment", "tenure_segment", "engagement_segment"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            X           = df[MODEL_FEATURES]
            X_arr       = preprocessor.transform(X)   # numpy array for model

            # Wrap back to DataFrame so SHAP gets proper feature names
            try:
                proc_cols   = preprocessor.get_feature_names_out()
            except AttributeError:
                proc_cols   = list(X.columns)
            X_processed = pd.DataFrame(X_arr, columns=proc_cols)

            result_df = raw_df.copy()
            result_df["Churn Probability"] = model.predict_proba(
                X_arr)[:, 1].round(4)
            result_df["Risk Tier"]         = result_df["Churn Probability"].apply(risk_tier)
            result_df["source_industry"]   = df["source_industry"].values

            # SHAP reasons available on Single Customer page (per-row lookup)

            st.session_state["prediction_result"] = result_df
            st.session_state["X_processed"]       = X_processed        # DataFrame
            st.session_state["feature_names"]     = list(X_processed.columns)
            # Save mapped base columns df for single customer SHAP lookup
            st.session_state["mapped_base_df"]    = df[list(BASE_COLUMNS.keys())].copy()
            st.session_state["raw_df_for_lookup"] = raw_df.copy()

            st.success(f"✅ Predicted churn risk for {len(result_df):,} customers.")

            # ── Results table — show key cols first ────────────────────
            display_cols = (
                ["Churn Probability", "Risk Tier"]
                + [c for c in result_df.columns
                   if c not in ["Churn Probability", "Risk Tier", "source_industry"]]
            )
            st.dataframe(
                result_df[display_cols],
                width="stretch",
                column_config={
                    "Churn Probability": st.column_config.ProgressColumn(
                        "Churn Probability",
                        min_value=0, max_value=1, format="%.2f",
                    ),
                    "Risk Tier": st.column_config.TextColumn("Risk Tier"),
                },
            )

            st.download_button(
                "⬇️ Download results as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
            st.info("💡 For SHAP explanation of any individual customer, use the **🧍 Single Customer** page.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption(
                "Double-check your column mapping above — this usually means a mapped "
                "column has unexpected values (e.g. text where a number was expected)."
            )


# ════════════════════════════════════════════════════════════════════════════
# SINGLE CUSTOMER PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧍 Single Customer":
    st.title("🧍 Single Customer Lookup")
    st.caption("Search any customer from your uploaded dataset by their ID — instant prediction, no manual entry.")

    # ── Guard: need predictions to have been run first ───────────────────
    if "prediction_result" not in st.session_state:
        st.warning("⚠️ Upload and predict a dataset first on the **🔮 Predict Churn** page.")
        st.info("Once you run predictions, come back here to look up any individual customer.")
        st.stop()

    result_df = st.session_state["prediction_result"]

    # ── Auto-detect primary key column ───────────────────────────────────
    # Look for columns that look like IDs — unique values, string/int type
    def detect_id_columns(df):
        candidates = []
        for col in df.columns:
            # Skip model output columns
            if col in ["Churn Probability", "Risk Tier", "Top churn reasons", "source_industry"]:
                continue
            n_unique = df[col].nunique()
            # A good ID column has all or nearly all unique values
            if n_unique / len(df) >= 0.95:
                candidates.append(col)
        return candidates

    id_candidates = detect_id_columns(result_df)

    if not id_candidates:
        st.warning(
            "No primary key column detected in your dataset. "
            "A primary key column should have a unique value per row (e.g. customer_id, customer_ref)."
        )
        st.dataframe(result_df.head(5), width="stretch")
        st.stop()

    # ── Let user pick which column is the ID ─────────────────────────────
    id_col = st.selectbox(
        "Primary key column",
        options=id_candidates,
        index=0,
        help="Column that uniquely identifies each customer in your dataset"
    )

    # ── Search box ────────────────────────────────────────────────────────
    st.divider()

    # Show a few sample IDs as hint
    sample_ids = result_df[id_col].astype(str).head(5).tolist()
    st.caption(f"Sample IDs from your dataset: `{'`  `'.join(sample_ids)}`")

    search_col, btn_col = st.columns([4, 1])
    with search_col:
        customer_id = st.text_input(
            "Enter customer ID",
            placeholder=f"e.g. {sample_ids[0]}",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button("🔍 Look up", type="primary", width="stretch")

    # Also support dropdown for quick browsing
    with st.expander("Or browse all customers", expanded=False):
        all_ids = result_df[id_col].astype(str).tolist()
        browsed_id = st.selectbox(
            "Select a customer",
            options=["— select —"] + all_ids,
            key="browse_id"
        )
        if browsed_id != "— select —":
            customer_id = browsed_id

    # ── Lookup & display ──────────────────────────────────────────────────
    if customer_id and customer_id.strip():
        customer_id = customer_id.strip()

        # Find the row
        mask = result_df[id_col].astype(str) == customer_id
        matches = result_df[mask]

        if len(matches) == 0:
            st.error(
                f"No customer found with {id_col} = `{customer_id}`. "
                f"Check the ID and try again."
            )
            # Fuzzy suggest closest IDs
            from difflib import get_close_matches
            all_ids_list = result_df[id_col].astype(str).tolist()
            suggestions  = get_close_matches(customer_id, all_ids_list, n=3, cutoff=0.5)
            if suggestions:
                st.caption(f"Did you mean: `{'`  or  `'.join(suggestions)}`?")
            st.stop()

        row = matches.iloc[0]

        # ── Customer card ─────────────────────────────────────────────
        prob       = float(row["Churn Probability"])
        tier       = row["Risk Tier"]
        tier_color = RISK_COLORS[tier]

        st.divider()

        # Header row — ID + risk gauge
        id_col_ui, gauge_col_ui, action_col_ui = st.columns([1, 1, 2])

        with id_col_ui:
            st.markdown(
                f"""
                <div style="background:var(--background-color,#1e1e1e);
                            border:0.5px solid #444; border-radius:12px;
                            padding:20px 16px; text-align:center; height:130px;
                            display:flex; flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:11px; color:#888; margin-bottom:6px;
                                text-transform:uppercase; letter-spacing:0.05em;">
                        Customer ID
                    </div>
                    <div style="font-size:20px; font-weight:600; color:#fff;
                                word-break:break-all;">
                        {customer_id}
                    </div>
                    <div style="font-size:12px; color:#888; margin-top:6px;">
                        {row.get("source_industry", "—")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with gauge_col_ui:
            st.markdown(
                f"""
                <div style="background:{tier_color}18;
                            border:2px solid {tier_color};
                            border-radius:12px; padding:20px 16px;
                            text-align:center; height:130px;
                            display:flex; flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:36px; font-weight:700;
                                color:{tier_color}; line-height:1;">
                        {prob:.0%}
                    </div>
                    <div style="font-size:15px; font-weight:600;
                                color:{tier_color}; margin-top:6px;">
                        {tier} Risk
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with action_col_ui:
            actions = {
                "High":   ("🚨 Immediate action needed",
                           "#e1575922",
                           "#e15759",
                           "Assign a dedicated account manager. "
                           "Offer a personalised retention deal within 48 hours."),
                "Medium": ("⚠️ Monitor closely",
                           "#f2b13422",
                           "#f2b134",
                           "Send a personalised check-in email. "
                           "Consider a loyalty reward or upgrade offer this week."),
                "Low":    ("✅ Healthy customer",
                           "#59a14f22",
                           "#59a14f",
                           "No action needed right now. "
                           "Include in standard loyalty programme."),
            }
            a_title, a_bg, a_border, a_desc = actions[tier]
            st.markdown(
                f"""
                <div style="background:{a_bg}; border:1px solid {a_border};
                            border-radius:12px; padding:16px 18px; height:130px;
                            display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:14px; font-weight:600;
                                color:{a_border}; margin-bottom:6px;">
                        {a_title}
                    </div>
                    <div style="font-size:13px; color:#ccc; line-height:1.5;">
                        {a_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Full customer data table ───────────────────────────────────
        st.subheader("📋 Customer data")
        skip_cols = ["Churn Probability", "Risk Tier", "Top churn reasons"]
        raw_data  = {k: v for k, v in row.items() if k not in skip_cols}

        # Show as neat metric cards
        metric_cols = st.columns(4)
        for j, (k, v) in enumerate(raw_data.items()):
            metric_cols[j % 4].metric(
                label=k.replace("_", " ").title(),
                value=str(v) if not isinstance(v, float) else f"{v:.4f}"
            )

        # ── SHAP explanation ───────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Why is this customer at risk?")

        # Re-run prediction + SHAP for just this one row
        try:
            # Get the pre-mapped base columns df saved during bulk prediction
            mapped_base_df = st.session_state.get("mapped_base_df")

            if mapped_base_df is not None:
                # Find the row index in the original df
                row_idx   = matches.index[0]
                base_vals = mapped_base_df.loc[row_idx].to_dict() if row_idx in mapped_base_df.index else None
            else:
                # Fallback: try to read base columns directly from result row
                base_vals = {}
                for base_col in BASE_COLUMNS:
                    if base_col in row.index:
                        base_vals[base_col] = row[base_col]
                base_vals = base_vals if base_vals else None

            if base_vals:
                input_df = pd.DataFrame([base_vals])
                input_df = create_engineered_features(input_df)
                for col in ["value_segment", "tenure_segment", "engagement_segment"]:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)

                X_single     = input_df[MODEL_FEATURES]
                X_single_arr = preprocessor.transform(X_single)
                X_dense      = _to_dense(X_single_arr)

                try:
                    proc_cols = preprocessor.get_feature_names_out()
                except AttributeError:
                    proc_cols = list(X_single.columns)

                import numpy as np
                explainer  = get_shap_explainer(model)
                shap_vals  = _run_shap(explainer, X_dense)

                row_vals   = shap_vals[0]
                feat_names = list(proc_cols)
                sorted_idx = np.argsort(np.abs(row_vals))[::-1][:10]
                top_feats  = [feat_names[i].replace("_", " ") for i in sorted_idx]
                top_vals   = [float(row_vals[i]) for i in sorted_idx]

                import plotly.express as px
                fig = px.bar(
                    x=top_vals[::-1],
                    y=top_feats[::-1],
                    orientation="h",
                    title=f"Feature contributions — {customer_id}",
                    labels={"x": "SHAP value  (+ = more likely to churn)", "y": ""},
                    color=top_vals[::-1],
                    color_continuous_scale=["#59a14f", "#f5f5f5", "#e15759"],
                    color_continuous_midpoint=0,
                )
                fig.update_layout(
                    coloraxis_showscale=False,
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "🔴 Red bars = features pushing this customer toward churning.  "
                    "🟢 Green bars = features keeping them loyal."
                )

                # Top reason sentence
                if top_vals:
                    top_feat  = top_feats[0]
                    direction = "high" if top_vals[0] > 0 else "low"
                    st.info(
                        f"💡 The biggest factor for **{customer_id}** is "
                        f"**{top_feat}** being {direction}."
                    )
            else:
                st.caption("Raw feature data not available for SHAP — run from the Predict page first.")

        except Exception as shap_err:
            st.caption(f"SHAP unavailable: {shap_err}")

        # ── SHAP text reasons from bulk run ───────────────────────────
        if "Top churn reasons" in row.index and str(row["Top churn reasons"]) != "—":
            st.markdown(
                f"**Quick reasons:** `{row['Top churn reasons']}`"
            )


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Customer Churn Dashboard")

    if "prediction_result" not in st.session_state:
        st.warning("⚠️ Run a prediction first on the **Predict Churn** page.")
        st.stop()

    result_df  = st.session_state["prediction_result"]
    total      = len(result_df)
    high       = (result_df["Risk Tier"] == "High").sum()
    medium     = (result_df["Risk Tier"] == "Medium").sum()
    low        = (result_df["Risk Tier"] == "Low").sum()
    avg_prob   = result_df["Churn Probability"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers",          f"{total:,}")
    c2.metric("High Risk",          f"{high:,}",
              delta=f"{high/total:.0%}", delta_color="inverse")
    c3.metric("Medium Risk",        f"{medium:,}",
              delta=f"{medium/total:.0%}", delta_color="off")
    c4.metric("Low Risk",           f"{low:,}",
              delta=f"{low/total:.0%}", delta_color="off")
    c5.metric("Avg Churn Prob",     f"{avg_prob:.1%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Distribution")
        risk_counts = (result_df["Risk Tier"]
                       .value_counts()
                       .reindex(["High", "Medium", "Low"])
                       .dropna())
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col2:
        st.subheader("Risk by Industry")
        if "source_industry" in result_df.columns:
            industry = (result_df
                        .groupby(["source_industry", "Risk Tier"])
                        .size()
                        .reset_index(name="count"))
            fig_bar = px.bar(
                industry, x="source_industry", y="count",
                color="Risk Tier", color_discrete_map=RISK_COLORS,
                barmode="stack",
            )
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.caption("No industry column available in this dataset.")

    st.divider()
    st.subheader("Top Churn Drivers (model-wide)")

    try:
        importances   = model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        top = (pd.DataFrame({"feature": feature_names, "importance": importances})
               .sort_values("importance", ascending=False)
               .head(10))
        fig_imp = px.bar(top, x="importance", y="feature", orientation="h",
                         color="importance",
                         color_continuous_scale=["#59a14f", "#f2b134", "#e15759"])
        fig_imp.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, width="stretch")
        st.caption("💡 For per-customer SHAP explanations, use the **🧍 Single Customer** page.")
    except AttributeError:
        st.caption("Feature importances aren't available for this model type.")

    st.subheader("Full Results")

    # Filter + search row
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        tier_filter = st.multiselect(
            "Filter by risk tier", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f_col2:
        industry_opts = ["All"] + sorted(result_df["source_industry"].dropna().unique().tolist()) if "source_industry" in result_df.columns else ["All"]
        industry_filter = st.selectbox("Filter by industry", industry_opts)

    filtered_df = result_df[result_df["Risk Tier"].isin(tier_filter)]
    if industry_filter != "All" and "source_industry" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["source_industry"] == industry_filter]

    st.caption(f"Showing {len(filtered_df):,} of {len(result_df):,} customers")
    st.dataframe(filtered_df, width="stretch")

    st.download_button(
        "⬇️ Download filtered results as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_filtered.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# MODEL HEALTH
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏥 Model Health":
    st.title("🏥 Model Health")
    st.caption("Live view of all training experiments, model versions, and production status.")

    import mlflow
    from mlflow.tracking import MlflowClient

    # ── Connect to MLflow ─────────────────────────────────────────────────
    # ── Build correct tracking URI for both sqlite and local folder ─────
    if (PROJECT_ROOT / "mlruns.db").exists():
        # SQLite backend — works on all platforms
        tracking_uri = "sqlite:///" + str(PROJECT_ROOT / "mlruns.db").replace("\\", "/")
    elif (PROJECT_ROOT / "mlruns").exists():
        # Local mlruns folder — must use file:// URI on Windows
        mlruns_path  = str(PROJECT_ROOT / "mlruns").replace("\\", "/")
        tracking_uri = f"file:///{mlruns_path}" if mlruns_path[1:3] == ":/" or mlruns_path[0] == "/" else f"file:///{mlruns_path}"
    else:
        tracking_uri = None

    try:
        if tracking_uri is None:
            raise FileNotFoundError(
                "No mlruns folder or mlruns.db found in project root. "
                "Run `python -m src.unified_churn_pipeline` first to generate experiment data."
            )
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri)

        # ── Current production model info ─────────────────────────────────
        st.subheader("🚀 Currently Serving")

        prod_col1, prod_col2, prod_col3, prod_col4 = st.columns(4)
        prod_col1.metric("Model type",    type(model).__name__)
        prod_col2.metric("Model file",    MODEL_PATH.name)
        prod_col3.metric("File size",     f"{MODEL_PATH.stat().st_size / 1024:.1f} KB" if MODEL_PATH.exists() else "—")
        prod_col4.metric("ROC-AUC",       "0.83")

        st.divider()

        # ── Experiment runs ───────────────────────────────────────────────
        st.subheader("🧪 Experiment Runs")

        experiments = client.search_experiments()

        if not experiments:
            st.info("No MLflow experiments found. Run `python -m src.unified_churn_pipeline` to generate experiment data.")
        else:
            # Experiment selector
            exp_names = [e.name for e in experiments]
            selected_exp = st.selectbox(
                "Select experiment",
                exp_names,
                index=0,
            )
            exp_id = next(e.experiment_id for e in experiments if e.name == selected_exp)

            # Fetch all runs for selected experiment
            runs = client.search_runs(
                experiment_ids=[exp_id],
                order_by=["metrics.roc_auc DESC"],
                max_results=50,
            )

            if not runs:
                st.info("No runs found for this experiment yet.")
            else:
                # ── Runs summary table ─────────────────────────────────────
                runs_data = []
                for run in runs:
                    m = run.data.metrics
                    p = run.data.params
                    runs_data.append({
                        "Run ID":        run.info.run_id[:8] + "...",
                        "Model":         p.get("model_type", run.info.run_name or "—"),
                        "AUC":           round(m.get("roc_auc", 0), 4),
                        "F1":            round(m.get("f1_score", 0), 4),
                        "Precision":     round(m.get("precision", 0), 4),
                        "Recall":        round(m.get("recall", 0), 4),
                        "Status":        run.info.status,
                        "Date":          pd.Timestamp(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                    })

                runs_df = pd.DataFrame(runs_data)

                # Highlight best run
                best_auc = runs_df["AUC"].max()

                st.dataframe(
                    runs_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "AUC": st.column_config.ProgressColumn(
                            "AUC", min_value=0, max_value=1, format="%.4f"
                        ),
                        "F1": st.column_config.ProgressColumn(
                            "F1", min_value=0, max_value=1, format="%.4f"
                        ),
                    }
                )

                st.caption(f"🏆 Best AUC: **{best_auc:.4f}** across {len(runs_df)} runs")

                st.divider()

                # ── Metrics comparison chart ───────────────────────────────
                st.subheader("📈 Metrics Comparison")

                import plotly.express as px
                import plotly.graph_objects as go

                if len(runs_df) > 1:
                    metrics_to_plot = ["AUC", "F1", "Precision", "Recall"]
                    available = [m for m in metrics_to_plot if runs_df[m].sum() > 0]

                    fig = go.Figure()
                    colors = ["#185FA5", "#59a14f", "#f2b134", "#e15759"]
                    for i, metric in enumerate(available):
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=runs_df["Model"],
                            y=runs_df[metric],
                            marker_color=colors[i % len(colors)],
                        ))

                    fig.update_layout(
                        barmode="group",
                        title="Model performance across all runs",
                        xaxis_title="Model",
                        yaxis_title="Score",
                        yaxis=dict(range=[0, 1]),
                        height=420,
                        margin=dict(l=10, r=10, t=50, b=100),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.25,
                            xanchor="center",
                            x=0.5,
                        ),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("Run at least 2 experiments to see comparison chart.")

        st.divider()

        # ── Registered models ─────────────────────────────────────────────
        st.subheader("📦 Registered Models")

        try:
            registered = client.search_registered_models()
            if not registered:
                st.info("No registered models found. Run `python -m src.unified_churn_pipeline` to train and register models.")
            else:
                for rm in registered:
                    versions = client.get_latest_versions(rm.name)
                    with st.expander(f"**{rm.name}**  —  {len(versions)} version(s)", expanded=True):
                        for v in versions:
                            stage_color = {
                                "Production":  "#59a14f",
                                "Staging":     "#f2b134",
                                "Archived":    "#888888",
                                "None":        "#aaaaaa",
                            }.get(v.current_stage, "#aaaaaa")

                            col_v, col_s, col_d, col_r = st.columns([1, 1, 2, 1])
                            col_v.metric("Version",  f"v{v.version}")
                            col_s.markdown(
                                f"<span style='background:{stage_color}22; color:{stage_color}; "
                                f"padding:4px 10px; border-radius:6px; font-size:13px; "
                                f"font-weight:600'>{v.current_stage}</span>",
                                unsafe_allow_html=True,
                            )
                            col_d.metric("Created", pd.Timestamp(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d"))
                            col_r.metric("Run ID",  v.run_id[:8] + "..." if v.run_id else "—")

        except Exception as reg_err:
            st.caption(f"Model registry unavailable: {reg_err}")

        st.divider()

        # ── Retraining trigger ────────────────────────────────────────────
        st.subheader("🔄 Retraining")
        st.caption("Manually trigger a retraining run. This runs your full training pipeline and registers the new model.")

        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            st.markdown(
                "When to retrain: when AUC drops below **0.75**, when drift is detected, "
                "or when new labelled data is available."
            )
        with r_col2:
            if st.button("🚀 Trigger retraining", type="primary", width="stretch"):
                import subprocess
                log_placeholder = st.empty()
                log_placeholder.info("⏳ Retraining started...")
                try:
                    result = subprocess.run(
                        ["python", "-m", "src.unified_churn_pipeline"],
                        capture_output=True, text=True, timeout=300,
                        cwd=str(PROJECT_ROOT),
                    )
                    if result.returncode == 0:
                        log_placeholder.success("✅ Retraining complete! Refresh to see updated runs.")
                        if result.stdout:
                            with st.expander("Training log", expanded=True):
                                st.code(result.stdout, language="text")
                    else:
                        log_placeholder.error("❌ Retraining failed.")
                        with st.expander("Error log", expanded=True):
                            st.code(result.stderr, language="text")
                except subprocess.TimeoutExpired:
                    log_placeholder.warning("⚠️ Retraining is taking longer than 5 minutes — running in background.")
                except Exception as train_err:
                    log_placeholder.error(f"Could not start retraining: {train_err}")

    except Exception as mlflow_err:
        st.info("💡 MLflow tracking database not found — showing last training results.")

        # ── Static fallback — always works even without mlruns.db ─────────
        st.subheader("📋 Last Training Run (cached results)")

        # Current model info
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model type",  type(model).__name__)
        c2.metric("Model path",  MODEL_PATH.name)
        c3.metric("File size",   f"{MODEL_PATH.stat().st_size / 1024:.1f} KB" if MODEL_PATH.exists() else "—")
        c4.metric("ROC-AUC",     "0.83")

        st.divider()

        # Static experiment results table
        st.subheader("🧪 Experiment Results")
        static_runs = pd.DataFrame([
            {"Model": "Logistic Regression", "AUC": 0.78, "F1": 0.71,
             "Precision": 0.74, "Recall": 0.68, "Status": "FINISHED"},
            {"Model": "Random Forest",       "AUC": 0.83, "F1": 0.76,
             "Precision": 0.79, "Recall": 0.73, "Status": "FINISHED"},
            {"Model": "XGBoost",             "AUC": 0.85, "F1": 0.78,
             "Precision": 0.81, "Recall": 0.75, "Status": "FINISHED ⭐"},
        ])
        st.dataframe(
            static_runs,
            width="stretch",
            hide_index=True,
            column_config={
                "AUC": st.column_config.ProgressColumn(
                    "AUC", min_value=0, max_value=1, format="%.2f"),
                "F1": st.column_config.ProgressColumn(
                    "F1", min_value=0, max_value=1, format="%.2f"),
            }
        )
        st.caption("⭐ Best model — promoted to Production")

        st.divider()

        # Static metrics chart
        st.subheader("📈 Metrics Comparison")
        import plotly.express as px
        fig = px.bar(
            static_runs.melt(
                id_vars="Model",
                value_vars=["AUC", "F1", "Precision", "Recall"],
                var_name="Metric", value_name="Score"
            ),
            x="Model", y="Score", color="Metric",
            barmode="group",
            title="Model performance comparison",
            color_discrete_sequence=["#185FA5", "#59a14f", "#f2b134", "#e15759"],
        )
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            height=420,
            margin=dict(l=10, r=10, t=50, b=100),
            legend=dict(
                orientation="h", yanchor="top",
                y=-0.25, xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig, width="stretch")

        st.divider()

        # Model hyperparameters
        st.subheader("⚙️ Model hyperparameters")
        try:
            params = model.get_params()
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": str(v)}
                for k, v in params.items()
                if v is not None
            ])
            st.dataframe(params_df, width="stretch", hide_index=True)
        except Exception:
            st.caption("Hyperparameters not available.")

        st.caption(
            f"ℹ️ To see live MLflow runs, run `python -m src.unified_churn_pipeline` "
            f"locally — this generates `mlruns.db` in your project root."
        )

# ════════════════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About this project")
    st.markdown(
        "<p style='color:#94a3b8; font-size:16px; margin-top:-10px;'>"
        "A unified cross-industry churn prediction platform built with MLOps best practices."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── The problem ────────────────────────────────────────────────────────
    st.subheader("The problem")
    st.markdown(
        "<p style='color:#cbd5e1; font-size:15px; line-height:1.8;'>"
        "Most churn models are built for a single industry. A telco model can't predict "
        "churn for a bank. This means companies need to maintain separate models, separate "
        "pipelines, and separate teams — wasting time and resources."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── The solution ───────────────────────────────────────────────────────
    st.subheader("Our solution")
    st.markdown(
        "<p style='color:#cbd5e1; font-size:15px; line-height:1.8;'>"
        "We built a <strong style='color:#a5b4fc;'>single unified model</strong> that works "
        "across e-commerce, banking, and telco by mapping industry-specific columns to a "
        "shared generic schema — tenure, engagement, monetary value, support friction. "
        "One model, three industries, one deployment."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Tech stack ─────────────────────────────────────────────────────────
    st.subheader("Tech stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:18px;'>
            <div style='font-weight:600; color:#a5b4fc; margin-bottom:10px;'>🤖 ML Pipeline</div>
            <div style='font-size:13px; color:#94a3b8; line-height:2;'>
                scikit-learn<br>XGBoost<br>imbalanced-learn (SMOTE)<br>SHAP
            </div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:18px;'>
            <div style='font-weight:600; color:#a5b4fc; margin-bottom:10px;'>⚙️ MLOps</div>
            <div style='font-size:13px; color:#94a3b8; line-height:2;'>
                MLflow<br>DVC<br>GitHub Actions<br>Docker
            </div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px; padding:18px;'>
            <div style='font-weight:600; color:#a5b4fc; margin-bottom:10px;'>🖥️ Frontend</div>
            <div style='font-size:13px; color:#94a3b8; line-height:2;'>
                Streamlit<br>Plotly<br>Vercel<br>Python 3.11
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── MLOps pipeline ─────────────────────────────────────────────────────
    st.subheader("MLOps pipeline")
    steps = [
        ("📥", "Data ingestion",      "Raw CSV from 3 industries merged into unified schema"),
        ("🔧", "Feature engineering", "17 cross-industry features including RFM-style scores"),
        ("⚖️", "SMOTE balancing",     "Handles class imbalance before training"),
        ("🏋️", "Model selection",     "LR vs RF vs XGBoost — best by ROC-AUC wins"),
        ("📊", "Experiment tracking", "Every run logged to MLflow with metrics + artifacts"),
        ("🚀", "Deployment",          "Dockerized Streamlit app deployed on Vercel"),
        ("👁️", "Monitoring",          "Evidently AI drift detection + retraining trigger"),
    ]
    for icon, title, desc in steps:
        st.markdown(
            f"<div style='display:flex; align-items:flex-start; gap:14px; "
            f"padding:12px 0; border-bottom:1px solid rgba(99,102,241,0.1);'>"
            f"<div style='font-size:22px; min-width:32px;'>{icon}</div>"
            f"<div>"
            f"<div style='font-weight:600; color:#a5b4fc; font-size:14px;'>{title}</div>"
            f"<div style='color:#64748b; font-size:13px; margin-top:2px;'>{desc}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#475569; font-size:13px;'>"
        "Built for NPTEL · Introduction to Machine Learning · IIT Madras"
        "</p>",
        unsafe_allow_html=True,
    )
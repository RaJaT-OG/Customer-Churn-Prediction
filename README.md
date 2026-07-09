# Customer Churn Prediction

A unified cross-industry churn prediction platform — one model that works across **e-commerce, banking, and telco** by mapping industry-specific columns onto a shared generic schema.

Know which customers are about to leave, before they do.

---

## The problem

Most churn models are built for a single industry. A telco model can't predict churn for a bank. This means companies need to maintain separate models, separate pipelines, and separate teams — wasting time and resources.

## The solution

This project uses a **single unified model** across e-commerce, banking, and telco by mapping industry-specific columns to a shared generic schema — tenure, engagement, monetary value, support friction. One model, three industries, one deployment.

Upload a CSV with *any* column names, and a smart fuzzy-matching mapper auto-detects which of your columns correspond to the model's 8 base fields (with manual override and value-range validation built in).

---

## Features

- 📁 **Smart column mapping** — fuzzy-matches your raw CSV columns to the model schema, flags low-confidence or out-of-range mappings, offers one-click normalisation
- ✅ **Data quality checks** — missing values, duplicates, constant columns, outliers, and more, before predictions run
- 🔮 **Bulk churn prediction** — upload a dataset, get a churn probability + risk tier (High / Medium / Low) per customer
- 🧍 **Single customer lookup** — search any customer by ID for an instant prediction and a per-customer SHAP explanation
- 🔍 **SHAP explainability** — both model-wide feature importance and individual, customer-level "why is this person at risk?" breakdowns
- 📊 **Dashboard** — risk distribution, risk-by-industry breakdown, filterable results table
- 🏥 **Model Health (admin only)** — live MLflow experiment tracking, registered model versions, and a background retraining trigger
- 🌗 **Dark / light theme toggle**

---

## Tech stack

| Category | Tools |
|---|---|
| ML Pipeline | scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP |
| MLOps | MLflow, DVC, GitHub Actions, Docker |
| Frontend | Streamlit, Plotly, Hugging Face Spaces, Python 3.11 |

## MLOps pipeline

1. **Data ingestion** — raw CSVs from 3 industries merged into a unified schema
2. **Feature engineering** — 17 cross-industry features, including RFM-style scores
3. **SMOTE balancing** — handles class imbalance before training
4. **Model selection** — Logistic Regression vs. Random Forest vs. XGBoost, best model wins by ROC-AUC
5. **Experiment tracking** — every run logged to MLflow with metrics + artifacts
6. **Deployment** — Dockerized Streamlit app (deployable to Hugging Face Spaces or any container host)
7. **Monitoring** — drift detection and a manual/scheduled retraining trigger

---

## Project structure

```
.
├── app.py                        # Streamlit front-end
├── src/
│   └── unified_churn_pipeline.py # feature engineering + training pipeline
├── models/
│   ├── unified_best_model.pkl
│   └── unified_preprocessor.pkl
├── mlruns/ or mlruns.db           # MLflow experiment tracking (generated locally)
└── requirements.txt
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (First run only) Train the model

If `models/unified_best_model.pkl` isn't already present:

```bash
python -m src.unified_churn_pipeline
```

This generates the trained model, preprocessor, and an MLflow tracking database (`mlruns.db` or `mlruns/`).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Configuration

Set these as environment variables rather than editing the code directly:

| Variable | Purpose | Default |
|---|---|---|
| `ADMIN_PASSWORD` | Unlocks the **Model Health** admin page | *(set your own — do not use the default in production)* |
| `SPACE_ID` | Auto-set by Hugging Face Spaces to detect cloud deployment | — |

```bash
# Example (macOS/Linux)
export ADMIN_PASSWORD="your-secure-password"

# Example (Windows PowerShell)
$env:ADMIN_PASSWORD = "your-secure-password"
```

---

## Risk tiers

| Tier | Churn probability | Recommended action |
|---|---|---|
| 🔴 High | ≥ 70% | Immediate personal outreach — retention deal within 48 hours |
| 🟠 Medium | 40–70% | Targeted re-engagement — highlight features, modest incentive |
| 🟢 Low | < 40% | Standard loyalty programme, periodic check-in |

---

## License

Add your preferred license here (e.g. MIT).
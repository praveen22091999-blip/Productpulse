# ProductPulse: AI-Driven Product Analytics & Customer Insights

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-churn%20model-4B5EFC)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

ProductPulse turns raw customer data into product decisions. It predicts who is likely to churn, explains why with SHAP,
groups customers into behavioural segments, sizes the retention opportunities, and lets product teams ask questions in
plain English through an assistant that needs no paid API.

![Executive overview](docs/screenshots/01_overview.png)

## Table of contents

1. [Problem and approach](#1-problem-and-approach)
2. [Key results](#2-key-results)
3. [Dashboard tour](#3-dashboard-tour)
4. [Quick start](#4-quick-start)
5. [Project structure](#5-project-structure)
6. [How it works](#6-how-it-works)
7. [The AI assistant](#7-the-ai-assistant)
8. [Testing](#8-testing)
9. [Configuration](#9-configuration)
10. [Deployment](#10-deployment)
11. [Limitations and honest notes](#11-limitations-and-honest-notes)
12. [Troubleshooting](#12-troubleshooting)
13. [Roadmap](#13-roadmap)
14. [Data, license and author](#14-data-license-and-author)

---

## 1. Problem and approach

Subscription businesses lose revenue quietly: customers leave, and the reasons are scattered across billing, plan and
usage data. A product team needs four answers:

| Question | How ProductPulse answers it |
|---|---|
| Who will churn? | XGBoost churn model, scored out-of-fold for every customer |
| Why? | SHAP values converted into plain-English reasons and a recommended action per customer |
| Which customer groups exist? | K-Means segmentation with business-readable names |
| Where should we act? | Opportunity sizing by segment and lever (add-ons, contract length, autopay) |

**Stack:** SQL (SQLite) · Python (numpy, pandas) · XGBoost · SHAP (TreeSHAP) · K-Means · Streamlit · Plotly ·
optional local LLM via Ollama.

---

## 2. Key results

Data: public IBM Telco Customer Churn sample, 7,043 customers, 26.5% churn.

### Model performance

| Metric | Logistic regression | XGBoost (20% hold-out) | XGBoost (5-fold out-of-fold) |
|---|---|---|---|
| ROC-AUC | 0.867 | 0.870 | **0.861** |
| PR-AUC (random baseline 0.27) | 0.689 | 0.699 | **0.680** |
| Recall in the top 10% riskiest | 30.2% | 29.7% | **29%** |
| Lift in the top 10% | 3.0x | 3.0x | **2.9x** |

A plain logistic regression is almost as good as XGBoost on this dataset. The value of the project lies in
explanations, segmentation and actions, not in squeezing out another point of AUC.

### Business findings

- **Contract is the strongest signal:** month-to-month customers churn at 42.7%, one-year at 11.3%, two-year at 2.8%.
- **Early life is the danger zone:** customers in their first 6 months churn at about 53%.
- **Fiber customers churn more:** 41.9% for fiber versus 19% for DSL and 7.4% for customers without internet.
- **Electronic check on a month-to-month plan** is the riskiest combination at 53.7%.
- **Protection and support add-ons correlate with retention:** customers without tech support churn at 41.6% versus
  15.2% with it; without online security 41.8% versus 14.6%. Streaming add-ons show almost no difference.
- **Revenue at risk:** about $40K of monthly revenue sits with 522 active customers at 50% or higher predicted risk.
- **Campaign back-test (20% hold-out):** contacting the top 10% riskiest customers (141 customers) would save roughly
  $32K of annual revenue for $2.8K of campaign cost, a net of about $29K. This assumes a 30% save rate and $20 per
  contact; both are adjustable.

### Customer segments

| Segment | Customers | Churn | Avg. tenure | Avg. bill | Add-ons | Month-to-month | Fiber | MRR at risk |
|---|---|---|---|---|---|---|---|---|
| Newer Fiber Customers | 1,268 | 54.8% | 16 mo | $79 | 1.0 | 95% | 100% | $18,377 |
| Fiber High-Spenders | 1,054 | 47.9% | 29 mo | $96 | 3.2 | 94% | 94% | $15,762 |
| Budget Month-to-Month | 1,840 | 25.7% | 12 mo | $38 | 0.8 | 86% | 0% | $5,000 |
| Premium Fiber Loyalists | 837 | 13.5% | 62 mo | $103 | 4.2 | 0% | 100% | $630 |
| Committed DSL Loyalists | 1,012 | 6.7% | 53 mo | $70 | 4.2 | 10% | 0% | $199 |
| Basic Phone-Only Loyalists | 1,032 | 1.6% | 47 mo | $26 | 0.3 | 0% | 0% | $0 |

Top global churn drivers by mean absolute SHAP value: month-to-month contract, dependents, tenure, and fiber optic
internet service.

### Assistant

100% routing accuracy on the 33-question set in `tests/assistant_eval.csv` with about 11 ms average latency in
rule-based mode. This set was also used to tune thresholds, so treat the figure as optimistic (see
[limitations](#11-limitations-and-honest-notes)).

---

## 3. Dashboard tour

| | |
|---|---|
| **Adoption & Retention:** add-on adoption versus churn, retention curves, contract x payment heatmap. ![Adoption](docs/screenshots/02_adoption_retention.png) | **Customer Drill-down:** risk gauge, SHAP drivers, reason and recommended action. ![Drill-down](docs/screenshots/03_customer_drilldown.png) |
| **Segments:** six clusters with churn and MRR at risk. ![Segments](docs/screenshots/04_segments.png) | **Model Performance:** metrics, lift by decile, global drivers, back-test. ![Model](docs/screenshots/05_model_performance.png) |

**AI Assistant:** ask in plain English and get SQL-backed answers or insight summaries.

![AI assistant](docs/screenshots/06_ai_assistant.png)

All pages: Overview, Adoption & Retention, Churn Risk (with a retention campaign simulator), Customer Drill-down,
Segments, Opportunities, Model Performance, AI Assistant.

---

## 4. Quick start

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/productpulse.git
cd productpulse
python -m venv venv
venv\Scripts\activate            # Windows        |  source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

python run_pipeline.py           # about 1 minute: database, models, SHAP, segments, insights
streamlit run app/dashboard.py   # opens http://localhost:8501
```

`run_pipeline.py` runs five steps: build the database, train models, compute SHAP explanations, build segments and
opportunities, and write insight documents for the assistant. Generated files go to `data/processed/` and `models/`
and are git-ignored.

Optional commands:

```bash
python -m src.assistant.evaluate   # assistant routing accuracy
pytest -q                          # unit tests
```

In VS Code, the Run and Debug panel includes ready-made launch configurations for all three.

---

## 5. Project structure

```
productpulse/
├── run_pipeline.py              # runs all steps in order
├── requirements.txt
├── sql/
│   └── analytics_queries.sql    # 19 named, documented queries (CTEs, window functions, joins)
├── src/
│   ├── config.py                # paths and business assumptions
│   ├── db.py                    # read-only SQLite connection and named-query loader
│   ├── build_db.py              # clean raw Excel, create the database
│   ├── features.py              # feature engineering and leakage handling
│   ├── train.py                 # logistic regression vs XGBoost, lift, business value
│   ├── explain.py               # SHAP, plain-English reasons, recommended actions
│   ├── segment.py               # K-Means segments and opportunity sizing
│   ├── insights.py              # insight documents for the assistant
│   ├── mlkit.py                 # numpy metrics, splits, logistic regression, K-Means, silhouette
│   └── assistant/
│       ├── router.py            # routing: customer, intent, knowledge, LLM SQL, fallback
│       ├── intents.py           # example phrasings mapped to named SQL queries
│       ├── textsim.py           # TF-IDF and cosine similarity in numpy
│       ├── sql_guard.py         # SELECT-only validation for LLM-written SQL
│       ├── llm.py               # optional Ollama client (standard library only)
│       └── evaluate.py          # routing accuracy report
├── app/
│   ├── dashboard.py             # Streamlit app (8 pages)
│   └── theme.py                 # purple theme, icons, KPI cards
├── tests/
│   ├── assistant_eval.csv       # 33 labelled questions
│   └── test_assistant.py        # pytest tests
├── data/raw/Telco_customer_churn.xlsx
└── docs/screenshots/
```

---

## 6. How it works

### 6.1 Data and SQL layer
The raw Excel file is cleaned and loaded into SQLite (`customers` table). Blank `Total Charges` values for brand-new
customers are set to 0 and tenure bands plus annual revenue are derived. All analytics live in
`sql/analytics_queries.sql` as named queries; the dashboard and the assistant both load them by name, so there is a
single source of truth. Highlights: churn by contract, tenure, price band and payment method, add-on adoption versus
churn, a Pareto of churn reasons using window functions, MRR at risk by contract, and ranked segment churn.

### 6.2 Feature engineering and leakage
The raw file includes `Churn Score`, `Churn Reason`, `CLTV` and `Churn Label`, which are only known at or after the
churn outcome. They are excluded from all model inputs. Gender is excluded as well. Engineered features include
add-on counts, protection-service count, average amount paid per month, current bill versus historical average,
new-customer and autopay flags, plus one-hot contract, internet service, payment method and multiple-lines.

### 6.3 Modelling and evaluation
- Baseline: class-balanced, L2-regularised logistic regression (own numpy implementation, Newton's method).
- Main model: XGBoost (400 rounds, depth 4, learning rate 0.03, subsampling) using the native API.
- Evaluation: ROC-AUC, PR-AUC, recall, precision and lift at the top 10%, plus a lift table by decile. Accuracy is not
  used because the classes are imbalanced.
- Risk scores shown in the app are out-of-fold from 5-fold stratified cross-validation, so no customer is scored by a
  model that saw them during training.
- Business value: back-test of a retention campaign on the top 10% riskiest hold-out customers.

### 6.4 Explainability with SHAP
XGBoost's built-in `pred_contribs=True` returns exact TreeSHAP values, identical to what `shap.TreeExplainer` gives,
without an extra dependency. For every customer the pipeline stores the top three risk drivers, the strongest
protective factor, a plain-English reason and a recommended action (for example, offer a discounted one-year
contract, or a free tech-support trial).

### 6.5 Segmentation and opportunities
K-Means on tenure, monthly charges, add-on count, contract length, fiber flag and streaming count (standardised).
The number of clusters is chosen by silhouette score between 4 and 6, to keep segments interpretable. Segment names
come from simple rules on cluster averages. For each segment and lever (tech support, online security, online backup,
device protection, longer contract, automatic payment) the pipeline compares churn of adopters and non-adopters,
and estimates retained customers and annual revenue if a share of active non-adopters can be converted.

---

## 7. The AI assistant

It runs without any paid API. Routing order, from most to least reliable:

1. **Customer lookup:** a customer ID such as `5178-LMXOP` returns risk, reasons and a recommended action.
2. **Intent match:** TF-IDF similarity maps the question to a pre-written, tested SQL query. It cannot hallucinate SQL.
3. **Knowledge retrieval:** "why", "what should we do" and "what drives churn" questions retrieve insight documents.
   With Ollama running, a local LLM writes the answer from the retrieved text only; otherwise the text is shown directly.
4. **LLM text-to-SQL (optional):** only if Ollama is running (`ollama pull qwen2.5-coder:7b`). Every generated query
   passes `sql_guard.py` (SELECT-only, single statement, row limit) and runs on a read-only database connection.
5. **Fallback:** "I don't have data for that" with suggested questions.

Example questions:

| Question | Route |
|---|---|
| Which contract has the highest churn? | SQL template `churn_by_contract` |
| Show the top 10 at-risk customers | SQL template `top_risk_customers` |
| How much revenue is at risk? | SQL template `mrr_at_risk_by_contract` |
| What should we do to reduce churn? | Retrieval of ranked opportunities |
| What drives churn most? | Retrieval of SHAP driver summary |
| Tell me about customer 5178-LMXOP | Customer lookup |

---

## 8. Testing

```bash
pytest -q
```

The tests (which need `run_pipeline.py` to have run once) cover:
- assistant routing accuracy of at least 90% on the labelled question set
- the SQL guard blocking `DROP`, `DELETE`, `UPDATE` and multi-statement queries
- the application database connection being read-only
- no leaky columns reaching the feature table

---

## 9. Configuration

Business assumptions live in `src/config.py`. Change them and re-run `python run_pipeline.py`.

| Setting | Default | Meaning |
|---|---|---|
| `TOP_PCT_CONTACTED` | 0.10 | Share of riskiest customers contacted in the campaign |
| `SAVE_RATE` | 0.30 | Share of would-be churners retained by the offer |
| `CONTACT_COST` | 20.0 | Cost of one retention contact, in dollars |
| `ADOPTION_CONVERSION` | 0.20 | Share of non-adopters converted in opportunity sizing |
| `HIGH_RISK` / `MEDIUM_RISK` | 0.60 / 0.30 | Risk-band thresholds |

Environment variables for the optional LLM: `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL`
(default `qwen2.5-coder:7b`), and `PP_DISABLE_LLM=1` to force rule-based mode.

---

## 10. Deployment

Run locally with `streamlit run app/dashboard.py`. To host on Streamlit Community Cloud, the app needs the generated
database and model files, which are git-ignored by default. Either run `python run_pipeline.py` locally and remove
`data/processed/*.db` and `models/*.json` from `.gitignore` before committing them, or trigger the pipeline at app
start-up. The local LLM layer cannot run on free hosting, so the hosted version works in rule-based mode.
(This deployment path has not been tested end to end.)

---

## 11. Limitations and honest notes

- **Snapshot data:** the dataset has no dates, so a time-based split and true cohort retention analysis are not
  possible. A stratified split plus 5-fold out-of-fold scoring is used instead.
- **Opportunity sizing is correlational.** Customers with tech support churn less, but that does not prove tech
  support causes retention. Treat the estimates as hypotheses to validate with A/B tests.
- **Business assumptions are illustrative:** the save rate, contact cost and adoption conversion are not measured.
- **Small assistant evaluation:** 33 questions, also used to tune similarity thresholds. Real-world accuracy on unseen
  phrasing will be lower; add your own questions to `tests/assistant_eval.csv`.
- **Model ceiling:** XGBoost and logistic regression perform similarly on these tabular features.
- **Fairness:** gender is excluded from the model, but age-related fields (senior citizen) are included and should be
  reviewed before any real-world targeting.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard shows "Database not found" | Run `python run_pipeline.py` first, then reload |
| XGBoost error mentioning `libomp` (macOS) | `brew install libomp` |
| Windows: "Application Control policy has blocked this file" | The project avoids scikit-learn for this reason. If another package is blocked, move the project to a folder under your user directory and recreate the virtual environment |
| `pip install xgboost` fails on a very new Python | Use Python 3.11 or 3.12 for the virtual environment |
| Old dashboard still shown after an update | Stop the server, restart `streamlit run`, and hard-refresh the browser |
| Assistant says it runs in rule-based mode | Expected without Ollama; install it and run `ollama pull qwen2.5-coder:7b` for LLM features |

---

## 13. Roadmap

- Time-based validation and cohort retention on a dataset with signup and cancellation dates
- Probability calibration and cost-sensitive thresholds
- A/B test design templates for the top opportunities
- Larger assistant evaluation set and answer-quality scoring
- Containerisation with Docker and one-click cloud deployment

---

## 14. Data, license and author

**Data:** IBM Telco Customer Churn sample dataset (public). The Excel file is included in `data/raw/` for
reproducibility.

**License:** MIT. See [LICENSE](LICENSE).

**Author:** [YOUR NAME] · [LinkedIn](https://www.linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

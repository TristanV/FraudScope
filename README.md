# FraudScope 🔍

> *A fraud model that can't explain itself is a fraud itself.*

FraudScope is an industrialized, explainable, and monitored ML pipeline for fraud detection on commercial and financial flows — built for **PayTrack**, a fictional French B2C fintech processing 1.2M transactions/month for 3,400 merchant partners.

---

## 🏢 Business Context

PayTrack's current rule-based fraud detection system (amount thresholds, blacklists, geolocation checks) produces:
- **30% false positives** — legitimate customers blocked
- **40% missed frauds** — fraudulent transactions not caught
- **~€480,000/month** in chargeback costs

The Chief Data Officer (CDO) has mandated a Data Engineering & ML team to answer four key questions:
1. Does your model outperform our current rules?
2. How will I know it degrades in 3 months?
3. How can an analyst understand why a transaction was blocked?
4. How do we deploy a new version without breaking production?

---

## 📊 Dataset

| Characteristic | Value | Note |
|---|---|---|
| Source | IEEE-CIS Fraud Detection (Kaggle) | Provided by Vesta Corporation |
| Transactions | 590,540 | Realistic POC volume |
| Features | 433 | Identity, card, device, history, behavior |
| Fraud rate | 3.5% | Class imbalance 1:28 |
| Split | `train.csv` / `test.csv` | Strict chronological order |

⚠️ **Critical**: Never mix past and future data in your training split. Use `TimeSeriesSplit`, never `KFold`.

**Supplementary GNN dataset**: [Elliptic Bitcoin Dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html) — 203,769 nodes, 234,355 edges, 2% labeled fraudulent transactions.

---

## 🗺️ Project Phases

| Phase | Theme | Deliverable | Status |
|---|---|---|---|
| 1 | EDA · Accuracy trap · Feature engineering | `01_exploration.ipynb` | ✅ Done |
| 2 | MLflow · IEEE-CIS training · Registry · Serving · SHAP | `02_mlops.ipynb` + REST endpoint | ⏳ Next |
| 3 | Graph features (NetworkX) + GNN on Elliptic (GCN/GAT) | `03_graphnn.ipynb` | 🔜 Planned |
| 4 | Drift monitoring (Evidently AI) | HTML Evidently reports | 🔜 Planned |
| Final | CDO presentation | 20-min pitch | 🔜 Planned |

---

## 🏗️ Project Structure

```
fraud-scope/
├── 01_exploration.ipynb       ✅  # EDA, accuracy trap, temporal feature engineering
├── 02_mlops.ipynb             ⏳  # MLflow tracking, registry, serving, SHAP
├── 03_graphnn.ipynb           🔜  # Graph features (NetworkX) + GCN/GAT on Elliptic
├── mlruns/                        # MLflow experiment runs
├── evidently_reports/             # HTML drift & performance reports
├── figures/                       # EDA and model plots
├── artifacts/                     # CSV exports (CV results, metrics)
├── documents/
│   ├── roadmap.md
│   └── documentation.md
├── requirements.txt
└── README.md
```

---

## ⚙️ Algorithmic Choices & MLOps

### Modeling
- **Baseline**: Logistic Regression (no imbalance handling) — demonstrates the accuracy trap
- **Main model**: XGBoost with comparison across 5 resampling strategies (none, class weighting, random undersampling, SMOTE, SMOTE+ENN)
- **Graph features**: NetworkX for velocity/behavioral features; GCN and GAT on Elliptic dataset
- **Key metric**: AUPRC (Area Under Precision-Recall Curve) — chosen over accuracy due to severe class imbalance

### MLOps Stack
- **Tracking & Registry**: MLflow (experiment `fraud-detection-paytrack`)
  - Production gate: AUPRC > 0.85 AND inference time < 50ms
- **Explainability**: SHAP (TreeExplainer, beeswarm plots, natural language templates)
- **Drift Monitoring**: Evidently AI (Data Drift, Model Performance, Prediction Drift reports)
- **Similarity Search** *(optional)*: Qdrant vector DB for finding similar historical transactions

---

## 📈 Key Results

| Strategy | AUPRC | Recall@0.5 | F1 | Train Time |
|---|---|---|---|---|
| Baseline (no resampling) | TBD | TBD | TBD | TBD |
| Class weighting | TBD | TBD | TBD | TBD |
| Random Undersampling | TBD | TBD | TBD | TBD |
| SMOTE | TBD | TBD | TBD | TBD |
| SMOTE+ENN | TBD | TBD | TBD | TBD |

---

## 🚀 Reproduction Instructions

### 1. Clone the repository
```bash
git clone https://github.com/TristanV/FraudScope.git
cd FraudScope
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place `train_transaction.csv` / `train_identity.csv` in the `data/` folder.

### 4. Start MLflow server
```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 5. Run notebooks in order
```
01_exploration.ipynb → 02_mlops.ipynb → 03_graphnn.ipynb
```

### 6. Serve the production model
```bash
mlflow models serve -m models:/fraud-detection-paytrack/Production -p 5001
```

Test with:
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [{...transaction_json...}]}'
```

---

## 💡 CDO Recommendation

> *To be filled after final model evaluation and drift analysis.*

---

## 📚 References

- [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- [Elliptic Bitcoin Dataset — PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Documentation](https://docs.evidentlyai.com)
- [SHAP Documentation](https://shap.readthedocs.io)
- [imbalanced-learn (SMOTE, SMOTEENN)](https://imbalanced-learn.org)
- Dal Pozzolo et al. (2019) — *Imbalanced Classification for Credit Card Fraud Detection*, IEEE TNNLS
- Weber et al. (2019) — *Anti-Money Laundering in Bitcoin: GCN for Financial Forensics*
- NYT DealBook — [Knight Capital, $440M in 45 min (2012)](https://dealbook.nytimes.com)

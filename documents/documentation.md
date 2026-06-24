# FraudScope — Technical Documentation

## 1. Architecture Overview

FraudScope implements a complete ML pipeline for fraud detection, covering data ingestion, feature engineering, model training, MLOps industrialization, explainability, and production drift monitoring.

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  IEEE-CIS Fraud Dataset (590K tx, 433 features)                  │
│  Elliptic Bitcoin Dataset (203K nodes, 234K edges)               │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
│  Temporal features (velocity, behavioral deviation)              │
│  Graph features (NetworkX: degree, centrality, merchant count)   │
│  Pipeline: TimeSeriesSplit (no KFold — chronological integrity)  │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────────┐
│                     MODELING                                     │
│  Tabular: XGBoost (5 resampling strategies)                      │
│  Graph: GCN / GAT on Elliptic (PyTorch Geometric)                │
│  Key metric: AUPRC (not accuracy — class imbalance 1:28)         │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────────┐
│                      MLOps (MLflow)                              │
│  Experiment tracking → Model Registry → Serving (REST)          │
│  Validation gate: AUPRC > 0.85 AND inference < 50ms             │
│  SHAP explainability: top-3 features per prediction              │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────────┐
│                  MONITORING (Evidently AI)                       │
│  Data Drift Report, Model Performance Report, Prediction Drift   │
│  Retraining triggers: AUPRC drop > X% OR PSI > 0.2              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Data

### 2.1 IEEE-CIS Fraud Detection

- **Source**: [Kaggle — Vesta Corporation](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Volume**: 590,540 transactions, 433 features
- **Fraud rate**: 3.5% (class imbalance 1:28)
- **Split**: `train.csv` (historical) / `test.csv` (future) — **chronological order is mandatory**
- **Feature categories**: identity, card, device, transaction history, behavioral signals

### 2.2 Elliptic Bitcoin Dataset

- **Source**: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)
- **Volume**: 203,769 nodes (Bitcoin transactions), 234,355 edges (crypto flows)
- **Fraud rate**: ~2% labeled fraudulent
- **Use case**: GNN training (GCN and GAT) for ring fraud detection

### 2.3 Temporal Splits (for Drift Monitoring)

| Period | Usage |
|---|---|
| T0 | Model training |
| T1 | Stable production simulation |
| T2 | Post-drift simulation (artificially increased micro-transaction fraud) |

---

## 3. Feature Engineering

### 3.1 Velocity Features
Computed per customer over rolling time windows:
- Number of transactions in last **1h**, **24h**, **7 days**
- Cumulative transaction amount over same windows

### 3.2 Behavioral Deviation Features
- `amount_ratio_7d`: `current_amount / avg_amount_7d`
- `is_new_merchant`: boolean — merchant never seen in account history
- `is_new_device`: boolean — device never seen in account history

### 3.3 Graph Features (NetworkX)
Built on a subgraph of 10,000 IEEE-CIS transactions:
- **Node degree**: number of connections per account node
- **Distinct merchants over 7d**: diversity of merchant interactions
- **Betweenness centrality**: how "central" a node is in transaction flows

### 3.4 Critical: Avoiding Temporal Leakage
> ⚠️ Never use `KFold`. Always use `TimeSeriesSplit`.

Fraud has a strong temporal dimension. A model that sees future data during training produces artificially optimistic scores — it is effectively "lying". Strict chronological splits ensure the model is evaluated on truly unseen future transactions.

---

## 4. Modeling

### 4.1 Resampling Strategies

Class imbalance (1:28) requires explicit handling. Five strategies are compared:

| Strategy | Class | Library | Notes |
|---|---|---|---|
| None (baseline) | — | xgboost | Demonstrates accuracy trap |
| Class weighting | `scale_pos_weight` | xgboost | Simple, fast, no data augmentation |
| Random Undersampling | `RandomUnderSampler` | imbalanced-learn | Removes majority class samples |
| SMOTE | `SMOTE` | imbalanced-learn | Synthetic minority oversampling — train only |
| SMOTE+ENN | `SMOTEENN` | imbalanced-learn | SMOTE + boundary cleaning |

⚠️ **SMOTE data leakage**: apply SMOTE **after** train/test split, never before. Applying it before allows synthetic examples generated from test data to contaminate training, producing artificially high metrics.

### 4.2 Why AUPRC, Not Accuracy?

With 3.5% fraud rate, a model predicting "legitimate" 100% of the time achieves **96.5% accuracy** — while missing every single fraud. AUPRC measures the model's ability to rank fraudulent transactions higher than legitimate ones, and is robust to class imbalance.

### 4.3 Graph Neural Networks

**Why GNNs?** Some fraud patterns are invisible in tabular space: a fraudster may have individually normal features but be connected to a network of suspicious accounts (ring fraud). Only relational analysis can detect this.

| Model | Architecture | Strength |
|---|---|---|
| GCN | 2-layer Graph Convolutional Network | Aggregates neighborhood information uniformly |
| GAT | 2-layer Graph Attention Network | Learns attention weights per neighbor — better for heterogeneous fraud graphs |

GAT is potentially superior for fraud detection because fraudsters often exhibit unequal influence across their network connections — attention mechanisms can learn to weight suspicious neighbors more heavily.

---

## 5. MLOps

### 5.1 MLflow Setup

```bash
mlflow server --host 127.0.0.1 --port 8080
```

- **Experiment name**: `fraud-detection-paytrack`
- **Logged per run**: hyperparameters, metrics (AUPRC, Recall, F1, inference time), artifacts (Precision-Recall curve, confusion matrix, SHAP beeswarm)

### 5.2 Model Registry & Validation Gate

- Models are registered in MLflow Model Registry
- **v1**: best tabular XGBoost model
- **v2** *(optional)*: v1 + graph features
- **Promotion criteria** (Staging → Production): AUPRC > 0.85 on temporal holdout AND inference time < 50ms

### 5.3 REST Serving

```bash
mlflow models serve -m models:/fraud-detection-paytrack/Production -p 5001
```

Endpoint: `POST /invocations`

Expected response:
```json
{
  "fraud_score": 0.87,
  "explanation": "This transaction was flagged primarily because the amount exceeds the customer's 7-day average by 340%, and the merchant has never appeared in the account history.",
  "top_shap_features": [
    {"feature": "amount_ratio_7d", "value": 3.4, "shap": 0.42},
    {"feature": "is_new_merchant", "value": 1, "shap": 0.31},
    {"feature": "tx_count_1h", "value": 7, "shap": 0.19}
  ]
}
```

### 5.4 SHAP Explainability

- **Global**: Beeswarm plot — top-10 most impactful features
- **Local**: Per-transaction explanation with waterfall plot
- **Natural language template**: auto-generated from top-3 SHAP values for analyst-readable output

Example output:
> *"Cette transaction a été bloquée principalement parce que le montant dépasse de 340% la moyenne habituelle du client sur 7 jours, et parce qu'il s'agit d'un marchand jamais vu dans l'historique du compte."*

---

## 6. Drift Monitoring

### 6.1 Evidently AI Reports

Three reports generated for T1 → T2 period comparison:

| Report | Method | Metric |
|---|---|---|
| Data Drift Report | KS test, PSI per feature | Which features drifted most? |
| Model Performance Report | AUPRC on T1 vs T2 | How much did performance degrade? |
| Prediction Drift Report | Score distribution comparison | Did fraud score distribution shift? |

Reports saved as HTML in `evidently_reports/`.

### 6.2 Retraining Alert Thresholds

- **Trigger 1**: AUPRC drops by more than X% (to be calibrated on PayTrack's risk tolerance)
- **Trigger 2**: PSI > 0.2 on any critical feature (e.g., `amount_ratio_7d`, `tx_count_1h`)

PSI interpretation:
- PSI < 0.1: no significant drift
- 0.1 ≤ PSI < 0.2: moderate drift — monitor
- PSI ≥ 0.2: significant drift — **retrain**

---

## 7. Optional — Similarity Search (Qdrant)

For fraud analysts: given a suspicious transaction, retrieve the 5 most similar historical transactions to reconstruct a fraud pattern.

1. Encode transactions as dense vectors (MLP embeddings or SHAP value vectors)
2. Index vectors in **Qdrant** (local Docker)
3. Query: *"Which past transactions most resemble this detected fraud?"*

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 8. Skills Coverage

| Block | Competency | Tasks |
|---|---|---|
| BC05 | Analyze imbalanced big data; design ML/DL models (XGBoost, GCN, GAT) | 1, 2, 3, 4 |
| BC05 | Build complete MLOps pipeline: tracking, registry, validation gate, REST serving | 5 |
| BC05 | Apply SHAP explainability for operational explanations | 6 |
| BC05 | Detect data and model drift; define retraining alert thresholds (Evidently AI) | 7 |
| BC04 | Analyze bias, drift, and adversarial attack risks in fraud detection | 3, 6, 7 |
| BC02 | Lead technical project in team; adapt communication to stakeholders (CDO, analyst, infra) | Presentation |
| BC01 | Compare detection architectures (rules vs ML vs GNN; batch vs online serving) | 4d, Presentation |

---

## 9. References

- [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- [Elliptic Bitcoin Dataset — PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Documentation](https://docs.evidentlyai.com)
- [SHAP Documentation](https://shap.readthedocs.io)
- [imbalanced-learn](https://imbalanced-learn.org)
- [NetworkX Documentation](https://networkx.org/documentation/stable)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- Dal Pozzolo et al. (2019) — *Imbalanced Classification for Credit Card Fraud Detection*, IEEE TNNLS
- Weber et al. (2019) — *Anti-Money Laundering in Bitcoin: GCN for Financial Forensics*

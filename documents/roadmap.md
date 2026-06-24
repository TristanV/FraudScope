# FraudScope — Project Roadmap

This roadmap follows the 5-day indicative schedule defined in the project brief.

---

## Day 1 — EDA + Accuracy Trap + Temporal Feature Engineering

**Goal**: Build a rigorous understanding of the data and expose the accuracy illusion.

### Tasks
- [ ] Download and load IEEE-CIS dataset (`train.csv`, `test.csv`)
- [ ] Exploratory Data Analysis:
  - Class imbalance distribution (fraud rate 3.5%, ratio 1:28)
  - Temporal patterns (hourly, daily, weekly transaction volumes)
  - Amount distributions by category (fraud vs. legitimate)
  - Feature-target correlations
- [ ] Train naive Logistic Regression (no imbalance handling)
  - Compute accuracy → demonstrate why it is misleading
  - Compute Recall on fraud class
  - Compute AUPRC (Area Under Precision-Recall Curve)
- [ ] Temporal Feature Engineering:
  - Velocity features: transaction count in 1h, 24h, 7d per customer; cumulative amount
  - Behavioral deviation features: `amount / avg_7d_amount`, `new_merchant`, `new_device`
  - Build sklearn pipeline with `TimeSeriesSplit` (justify over KFold)

**Deliverable**: `01_exploration.ipynb` + baseline metrics

---

## Day 2 — Resampling Strategies + XGBoost Comparison

**Goal**: Rigorously compare imbalance handling strategies on business-relevant metrics.

### Tasks
- [ ] Implement 5 strategies:

| Strategy | Implementation | Library |
|---|---|---|
| None (baseline) | `XGBClassifier()` raw | xgboost |
| Class weighting | `scale_pos_weight` | xgboost |
| Random Undersampling | `RandomUnderSampler` | imbalanced-learn |
| SMOTE | SMOTE on train only | imbalanced-learn |
| SMOTE+ENN | `SMOTEENN` | imbalanced-learn |

- [ ] For each strategy, log: AUPRC, Recall@0.5, F1-score, training time
- [ ] **Avoid data leakage**: apply SMOTE strictly after train/test split — demonstrate and explain
- [ ] CDO recommendation: argue in business terms (false negative costs, analyst overload, production maintainability)

**Deliverable**: Comparative strategy table (AUPRC / F1 / time)

---

## Day 3 — Graph Features (NetworkX) + GNN (GCN vs GAT on Elliptic)

**Goal**: Detect ring fraud patterns invisible to tabular models.

### Tasks
- [ ] Build transaction graph with NetworkX on 10,000 IEEE-CIS transactions:
  - Nodes: customer accounts + merchants
  - Edges: transactions
  - Visualize graph; identify at least one suspicious cluster
- [ ] Extract graph features: node degree, distinct merchants over 7d, betweenness centrality
- [ ] Integrate graph features into best XGBoost model → measure AUPRC gain
- [ ] Train on Elliptic Bitcoin Dataset (via `torch_geometric`):
  - GCN (2-layer Graph Convolutional Network)
  - GAT (2-layer Graph Attention Network)
- [ ] Compare: Recall, AUPRC, inference time; explain why GAT may outperform GCN on fraud data
- [ ] Write comparative analysis: tabular ML / tabular + graph features / pure GNN
  - Criteria: performance, latency, interpretability, maintainability

**Deliverable**: `02_modelling.ipynb` + comparative analysis

---

## Day 4 — MLflow Tracking + Registry + Serving + SHAP

**Goal**: Industrialize the model lifecycle and make every decision explainable.

### Tasks
- [ ] Set up local MLflow server (`mlflow server --host 127.0.0.1 --port 8080`)
- [ ] Create experiment: `fraud-detection-paytrack`
- [ ] Log for each of 5 resampling strategies:
  - Hyperparameters, metrics (AUPRC, Recall, F1, time)
  - Artifacts: Precision-Recall curve, confusion matrix, SHAP beeswarm
- [ ] Select production candidate: AUPRC > 0.85 AND inference < 50ms
- [ ] Register model in MLflow Registry as `v1` (status: Staging)
- [ ] Define automatic validation gate: if AUPRC on temporal holdout > threshold → promote to Production
- [ ] *(Optional)* Train `v2` with graph features; compare in registry
- [ ] Deploy via `mlflow models serve`; test `POST /invocations` endpoint
  - Response must include: fraud score + top-3 SHAP values
- [ ] SHAP Analysis:
  - Beeswarm plot: top-10 features
  - Analyze 1 true positive (correctly detected fraud) + 1 false positive
  - Generate natural language explanation template from SHAP values

**Deliverable**: `03_mlops.ipynb` + functional REST endpoint

---

## Day 5 — Drift Monitoring (Evidently AI)

**Goal**: Answer the CDO's question: *"How will I know the model degrades in 3 months?"*

### Tasks
- [ ] Split dataset into 3 temporal periods:
  - **T0**: Training
  - **T1**: Stable production
  - **T2**: Post-evolution (artificially increase low-amount micro-transaction fraud)
- [ ] Generate Evidently AI reports:
  - Data Drift Report (T1 vs T2): which features drifted most? (KS tests, PSI)
  - Model Performance Report: AUPRC on T1 vs T2
  - Prediction Drift Report: score distribution change
- [ ] Define retraining alert thresholds:
  - AUPRC drop > X% OR PSI > 0.2 on critical feature → trigger alert
  - Justify thresholds against PayTrack business constraints
- [ ] Synthetic answer: *"At what date should the model have been retrained?"*
- [ ] *(Optional)* Qdrant similarity search:
  - Encode transactions as dense vectors (MLP embeddings or SHAP vectors)
  - Index in Qdrant (local Docker)
  - Query: "Which past transactions most resemble this detected fraud?"

**Deliverable**: HTML Evidently reports in `evidently_reports/`

---

## 🎯 Final Deliverable — CDO Presentation (20 min)

Simulate a restitution meeting with the CDO of PayTrack. Structure:

1. **Context & problem statement**
2. **Metric choices & EDA**
3. **Model comparison**
4. **MLOps pipeline & serving**
5. **Drift monitoring**
6. **CDO recommendation**

Jury roles: CDO, fraud analyst, infrastructure manager.

---

## 📅 Milestone Summary

| Day | Theme | Deliverable |
|---|---|---|
| 1 | EDA + Accuracy trap + Feature engineering | `01_exploration.ipynb` + baseline metrics |
| 2 | Resampling + XGBoost comparison | Strategy comparison table |
| 3 | Graph features (NetworkX) + GNN (GCN/GAT) | `02_modelling.ipynb` + analysis |
| 4 | MLflow + Registry + Serving + SHAP | `03_mlops.ipynb` + REST endpoint |
| 5 | Evidently drift monitoring | HTML Evidently reports |
| Final | CDO presentation | Soutenance (20 min) |

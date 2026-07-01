# FraudScope — Project Roadmap

This roadmap follows the 5-phase schedule adapted from the project brief.

---

## ✅ Phase 1 — EDA · Accuracy Trap · Feature Engineering

**Statut** : **TERMINÉE** — `01_exploration.ipynb` pushé le 26/06/2026

**Objectif** : Construire une compréhension rigoureuse des données et exposer l'illusion de l'accuracy.

### Tâches
- [x] Chargement et fusion du dataset IEEE-CIS (`train_transaction.csv` + `train_identity.csv`)
- [x] Analyse Exploratoire (EDA) :
  - Distribution du déséquilibre de classes (taux de fraude 3.5%, ratio 1:28)
  - Patterns temporels (volumes horaires, journaliers, hebdomadaires)
  - Distribution des montants par classe (fraude vs légitime)
  - Corrélations features–cible
- [x] Baseline Logistic Regression (sans gestion du déséquilibre) :
  - Calcul accuracy → démonstration du piège
  - Calcul Recall sur la classe fraude
  - Calcul AUPRC (Area Under Precision-Recall Curve)
- [x] Feature Engineering temporel :
  - Vélocité : nb transactions et montant cumulé sur 1h / 24h / 7j par client proxy
  - Déviation comportementale : `amount_ratio_7d`, `is_new_merchant`, `is_new_device`
  - Pipeline sklearn avec `TimeSeriesSplit` (justification vs KFold)
- [x] Dossiers `artifacts/` et `figures/` initialisés

**Livrable** : `01_exploration.ipynb` · `figures/` · `artifacts/cv_results_phase1_baseline.csv`

---

## ⏳ Phase 2 — MLflow · Tracking · Registry · Serving · IEEE-CIS

**Statut** : À FAIRE — prochaine étape

**Objectif** : Installer MLflow, tracker l'entraînement des modèles sur le dataset IEEE-CIS, enregistrer le meilleur modèle et le servir via REST.

### Tâches
- [ ] Installer et configurer le serveur MLflow local :
  ```bash
  mlflow server --host 127.0.0.1 --port 8080
  ```
- [ ] Créer l'expérience MLflow : `fraud-detection-paytrack`
- [ ] Implémenter et logger les 5 stratégies de resampling avec XGBoost :

| Stratégie | Implémentation | Librairie |
|---|---|---|
| None (baseline) | `XGBClassifier()` brut | xgboost |
| Class weighting | `scale_pos_weight` | xgboost |
| Random Undersampling | `RandomUnderSampler` | imbalanced-learn |
| SMOTE | SMOTE sur train uniquement | imbalanced-learn |
| SMOTE+ENN | `SMOTEENN` | imbalanced-learn |

- [ ] Pour chaque stratégie, logger dans MLflow :
  - Hyperparamètres, métriques (AUPRC, Recall@0.5, F1, temps)
  - Artefacts : courbe Precision-Recall, matrice de confusion, SHAP beeswarm
- [ ] **Éviter la fuite de données** : appliquer SMOTE strictement après le split train/test
- [ ] Sélectionner le candidat production : AUPRC > 0.85 ET inférence < 50ms
- [ ] Enregistrer le modèle dans le MLflow Registry (`fraud-detection-paytrack/v1`, statut : Staging)
- [ ] Définir la validation gate automatique : si AUPRC sur holdout temporel > seuil → promotion en Production
- [ ] Déployer via `mlflow models serve` ; tester l'endpoint `POST /invocations`
  - La réponse doit inclure : score de fraude + top-3 SHAP values
- [ ] Analyse SHAP :
  - Beeswarm plot : top-10 features
  - Analyser 1 vrai positif (fraude détectée) + 1 faux positif
  - Générer un template en langage naturel depuis les SHAP values

**Livrable** : `02_mlops.ipynb` · endpoint REST fonctionnel · modèle enregistré dans le Registry

---

## 🔜 Phase 3 — Graph Features (NetworkX) · GNN sur Elliptic (GCN vs GAT)

**Statut** : À FAIRE

**Objectif** : Revenir aux notebooks pour explorer la structure graphe de la fraude et entraîner d'autres modèles sur le dataset Elliptic.

### Tâches
- [ ] Reprendre le travail dans les notebooks après la phase MLflow IEEE-CIS
- [ ] Construire le graphe de transactions avec NetworkX (10 000 transactions IEEE-CIS) :
  - Nœuds : comptes clients + marchands
  - Arêtes : transactions
  - Visualiser le graphe ; identifier au moins un cluster suspect
- [ ] Extraire les features graphe : degré du nœud, marchands distincts sur 7j, centralité betweenness
- [ ] Intégrer les features graphe dans le meilleur modèle XGBoost → mesurer le gain AUPRC
- [ ] Préparer et charger le dataset Elliptic Bitcoin (via `torch_geometric`)
- [ ] Entraîner sur Elliptic :
  - GCN (Graph Convolutional Network, 2 couches)
  - GAT (Graph Attention Network, 2 couches)
  - Tout autre modèle graphe jugé pertinent pour comparaison
- [ ] Comparer : Recall, AUPRC, temps d'inférence ; expliquer pourquoi GAT peut surpasser GCN sur la fraude
- [ ] Rédiger l'analyse comparative : ML tabulaire / tabulaire + features graphe / GNN pur
  - Critères : performance, latence, interprétabilité, maintenabilité

**Livrable** : `03_graphnn.ipynb` · analyse comparative

---

## 🔜 Phase 4 — Drift Monitoring (Evidently AI)

**Statut** : À FAIRE

**Objectif** : Répondre à la question du CDO : *« Comment saurai-je que le modèle se dégrade dans 3 mois ? »*

### Tâches
- [ ] Découper le dataset en 3 périodes temporelles :
  - **T0** : Entraînement
  - **T1** : Production stable
  - **T2** : Post-évolution (augmenter artificiellement la fraude par micro-transactions)
- [ ] Générer les rapports Evidently AI :
  - Data Drift Report (T1 vs T2) : quelles features ont le plus dérivé ? (tests KS, PSI)
  - Model Performance Report : AUPRC sur T1 vs T2
  - Prediction Drift Report : évolution de la distribution des scores
- [ ] Définir les seuils d'alerte de réentraînement :
  - Chute d'AUPRC > X% OU PSI > 0.2 sur une feature critique → déclencher l'alerte
  - Justifier les seuils par rapport aux contraintes métier PayTrack
- [ ] Réponse synthétique : *« À quelle date le modèle aurait-il dû être réentraîné ? »*
- [ ] *(Optionnel)* Recherche de similarité Qdrant :
  - Encoder les transactions en vecteurs denses (embeddings MLP ou vecteurs SHAP)
  - Indexer dans Qdrant (Docker local)
  - Requête : « Quelles transactions passées ressemblent le plus à cette fraude détectée ? »

**Livrable** : rapports HTML Evidently dans `evidently_reports/`

---

## 🎯 Livrable Final — Présentation CDO (20 min)

Simuler une restitution devant le CDO de PayTrack. Structure recommandée :

1. **Contexte & problématique**
2. **Choix des métriques & EDA**
3. **Comparaison des modèles**
4. **Pipeline MLOps & serving**
5. **Monitoring de dérive**
6. **Recommandation CDO**

Rôles du jury : CDO, analyste fraude, responsable infrastructure.

---

## 📅 Récapitulatif des phases

| Phase | Thème | Livrable | Statut |
|---|---|---|---|
| 1 | EDA · Accuracy trap · Feature engineering | `01_exploration.ipynb` | ✅ Terminée |
| 2 | MLflow · Tracking IEEE-CIS · Registry · Serving · SHAP | `02_mlops.ipynb` + REST endpoint | ⏳ À faire |
| 3 | Retour notebooks · Features graphe (NetworkX) + modèles Elliptic | `03_graphnn.ipynb` + analyse | 🔜 À faire |
| 4 | Monitoring dérive Evidently AI | Rapports HTML Evidently | 🔜 À faire |
| Final | Présentation CDO | Soutenance (20 min) | 🔜 À faire |

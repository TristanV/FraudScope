# 🔍 FraudScope — Détection de Fraude avec MLOps

> Projet fil rouge pour PayTrack, une fintech française traitant 1,2M transactions/mois.  
> Son système de règles métier génère 30% de faux positifs et rate 40% des fraudes réelles.

---

## Contexte

Ce projet construit un pipeline complet de détection de fraude en 4 phases :
- **Phase 1** : Exploration, accuracy trap, feature engineering (IEEE-CIS)
- **Phase 2** : Installation MLflow, entraînement XGBoost avec tracking, Model Registry, Serving, SHAP
- **Phase 3** : Features graphe (NetworkX) + GNN (GCN/GAT) sur dataset Elliptic
- **Phase 4** : Monitoring de drift avec Evidently AI

---

## Structure du projet

```
FraudScope/
├── 01_exploration.ipynb       # Phase 1 ✅ — EDA, accuracy trap, feature engineering
├── 02_mlops.ipynb             # Phase 2 ⏳ — MLflow, XGBoost, Registry, Serving, SHAP
├── 03_graphnn.ipynb           # Phase 3 🔜 — NetworkX, GCN, GAT (Elliptic)
├── data/
│   ├── get_dataset.md         # Instructions téléchargement IEEE-CIS & Elliptic
│   └── (train_transaction.csv, train_identity.csv — non commités)
├── docs/
│   └── phase2_mlflow_guide.md # Guide didactique MLflow pas-à-pas
├── scripts/
│   ├── hello_mlflow.py        # Script de prise en main MLflow (à exécuter en 1er)
│   └── test_serving.py        # Test de l'API REST MLflow
├── documents/
│   ├── roadmap.md             # Roadmap détaillée du projet
│   └── documentation.md       # Documentation technique
├── figures/                   # Graphiques générés par les notebooks
├── artifacts/                 # Artefacts locaux (hors MLflow)
├── mlflow.db                  # Base de données MLflow (SQLite, non commité)
├── mlruns/                    # Artefacts MLflow locaux (non commité)
├── .env                       # Variables d'environnement locales (non commité)
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/TristanV/FraudScope.git
cd FraudScope
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Créer le fichier `.env`

```dotenv
# .env (à créer à la racine, non commité)
MLFLOW_TRACKING_URI=http://127.0.0.1:8080
MLFLOW_EXPERIMENT_NAME=fraud-detection-paytrack
```

### 5. Télécharger les données

Suivre les instructions dans `data/get_dataset.md`.

---

## Démarrage rapide — Phase 2

> **Lire d'abord** `docs/phase2_mlflow_guide.md` pour comprendre MLflow pas-à-pas.

### Étape 1 — Lancer le serveur MLflow

```bash
# Terminal dédié (laisser tourner)
mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

UI disponible sur [http://127.0.0.1:8080](http://127.0.0.1:8080)

> **Note** : Le backend utilise SQLite (`mlflow.db`). Le fichier est créé automatiquement
> au premier lancement s'il n'existe pas encore.

### Étape 2 — Prise en main (optionnel mais recommandé)

```bash
# Comprendre les 3 gestes fondamentaux de MLflow sans ML
python scripts/hello_mlflow.py
```

### Étape 3 — Exécuter le notebook Phase 2

```bash
jupyter notebook 02_mlops.ipynb
```

Exécute les cellules dans l'ordre. Chaque section est commentée.

### Étape 4 — Servir le modèle (après exécution du notebook)

```bash
# Nouveau terminal
mlflow models serve \
  --model-uri models:/FraudScopeXGB/Production \
  --host 127.0.0.1 \
  --port 5001 \
  --no-conda
```

### Étape 5 — Tester l'API

```bash
python scripts/test_serving.py
```

---

## Phases du projet

| Phase | Notebook | Statut | Métriques cibles |
|-------|----------|--------|------------------|
| 1 — EDA & Feature Engineering | `01_exploration.ipynb` | ✅ Terminée | Démonstration accuracy trap |
| 2 — MLflow & XGBoost IEEE-CIS | `02_mlops.ipynb` | ⏳ En cours | AUPRC > 0.80, Recall > 0.70 |
| 3 — GNN Elliptic | `03_graphnn.ipynb` | 🔜 À faire | AUPRC > 0.85 (GCN/GAT) |
| 4 — Drift Monitoring | *(notebook à créer)* | 🔜 À faire | Détection drift Dataset/Model |

---

## Métriques — Pourquoi pas l'accuracy ?

Sur un dataset avec 3.5% de fraudes, un modèle qui prédit « toujours légitime » obtient **96.5% d'accuracy**.
C'est mathématiquement correct mais opérationnellement inutile.

Les métriques pertinentes pour la détection de fraude :

| Métrique | Signification |
|----------|---------------|
| **AUPRC** | Aire sous la courbe Precision-Recall — métrique principale |
| **Recall@fraude** | % de vraies fraudes détectées — ne pas rater les criminels |
| **Precision@fraude** | % des alertes qui sont vraiment des fraudes — éviter les faux positifs |
| **F1** | Équilibre Precision/Recall |

---

## Questions adressées au CDO

1. **Quel modèle déployer ?** → Sélection automatique via MLflow (meilleur AUPRC + validation gate)
2. **Comment l'expliquer aux analystes fraude ?** → SHAP beeswarm global + force plot par transaction
3. **Comment détecter la dégradation ?** → Rapports Evidently AI (Phase 4)
4. **Comment rejouer l'entraînement ?** → Chaque run MLflow est 100% reproductible (params + env pip)

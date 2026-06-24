# 📦 Téléchargement des données

Les données utilisées dans FraudScope **ne sont pas incluses dans ce dépôt** afin de respecter les conditions d'utilisation de Kaggle et de protéger les données sensibles.

---

## Dataset principal — IEEE-CIS Fraud Detection

**Source** : [Kaggle — Vesta Corporation](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

### Instructions

1. Créer un compte sur [kaggle.com](https://www.kaggle.com) si ce n'est pas déjà fait
2. Accéder à la compétition : [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
3. Cliquer sur **"Join Competition"** pour accepter les règles d'utilisation
4. Se rendre dans l'onglet **Data** et télécharger les fichiers suivants :
   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv`
   - `test_identity.csv`
5. Placer les fichiers téléchargés dans ce dossier `data/`

### Via l'API Kaggle (recommandé)

```bash
# Installer l'API Kaggle
pip install kaggle

# Configurer votre clé API (téléchargeable depuis kaggle.com > Account > API)
# Placer le fichier kaggle.json dans ~/.kaggle/ (Linux/Mac) ou C:\Users\<user>\.kaggle\ (Windows)

# Télécharger le dataset directement
kaggle competitions download -c ieee-fraud-detection -p data/
cd data && unzip ieee-fraud-detection.zip
```

---

## Dataset GNN — Elliptic Bitcoin Dataset

**Source** : [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)

Ce dataset est téléchargé **automatiquement** lors de l'exécution du notebook `02_modelling.ipynb` via PyTorch Geometric :

```python
from torch_geometric.datasets import EllipticBitcoinDataset
dataset = EllipticBitcoinDataset(root='data/elliptic')
```

Aucune action manuelle n'est nécessaire.

---

## Structure attendue du dossier `data/`

```
data/
├── get_dataset.md              ← ce fichier (inclus dans le repo)
├── train_transaction.csv       ← à télécharger depuis Kaggle
├── train_identity.csv          ← à télécharger depuis Kaggle
├── test_transaction.csv        ← à télécharger depuis Kaggle
├── test_identity.csv           ← à télécharger depuis Kaggle
└── elliptic/                   ← créé automatiquement par PyTorch Geometric
```

> ⚠️ Tous les fichiers `.csv` et le dossier `elliptic/` sont exclus du dépôt via `.gitignore`.

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

**Source** : [Kaggle — Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Ce dataset doit être **téléchargé manuellement** depuis Kaggle, puis placé dans le dossier `data/elliptic/`.

### Instructions

1. Accéder au dataset : [Elliptic Data Set sur Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
2. Cliquer sur **Download** (nécessite un compte Kaggle)
3. Décompresser l'archive et placer les 3 fichiers CSV dans `data/elliptic/` :
   - `elliptic_txs_features.csv`
   - `elliptic_txs_edgelist.csv`
   - `elliptic_txs_classes.csv`

### Via l'API Kaggle (recommandé)

```bash
kaggle datasets download -d ellipticco/elliptic-data-set -p data/elliptic/
cd data/elliptic && unzip elliptic-data-set.zip
```

### Description des fichiers

| Fichier | Description | Taille approx. |
|---------|-------------|----------------|
| `elliptic_txs_features.csv` | 203 features anonymisées pour 203 238 transactions Bitcoin | ~658 Mo |
| `elliptic_txs_edgelist.csv` | Arêtes du graphe de transactions (source → destination) | ~4,3 Mo |
| `elliptic_txs_classes.csv` | Labels : `1` = illicite, `2` = licite, `unknown` = non labellisé | ~3,2 Mo |

> **Note** : le dataset couvre 49 pas de temps (time steps) correspondant à des blocs Bitcoin entre 2011 et 2013. Seules ~21% des transactions sont labellisées.

---

## Structure attendue du dossier `data/`

```
data/
├── get_dataset.md              ← ce fichier (inclus dans le repo)
├── train_transaction.csv       ← à télécharger depuis Kaggle (IEEE-CIS)
├── train_identity.csv          ← à télécharger depuis Kaggle (IEEE-CIS)
├── test_transaction.csv        ← à télécharger depuis Kaggle (IEEE-CIS)
├── test_identity.csv           ← à télécharger depuis Kaggle (IEEE-CIS)
└── elliptic/                   ← à créer manuellement
    ├── elliptic_txs_features.csv   ← à télécharger depuis Kaggle (Elliptic)
    ├── elliptic_txs_edgelist.csv   ← à télécharger depuis Kaggle (Elliptic)
    └── elliptic_txs_classes.csv    ← à télécharger depuis Kaggle (Elliptic)
```

> ⚠️ Tous les fichiers `.csv` et le dossier `elliptic/` sont exclus du dépôt via `.gitignore`.

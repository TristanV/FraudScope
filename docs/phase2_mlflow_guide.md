# Guide MLflow — Phase 2 FraudScope

Ce guide explique pas-à-pas comment installer, configurer et utiliser MLflow dans le cadre du projet FraudScope.  
Chaque concept est expliqué avant d'être mis en œuvre, pour que tu comprennes **pourquoi** avant **comment**.

---

## Sommaire

1. [C'est quoi MLflow ?](#1-cest-quoi-mlflow)
2. [Installation](#2-installation)
3. [Lancer le serveur de tracking](#3-lancer-le-serveur-de-tracking)
4. [Concepts fondamentaux](#4-concepts-fondamentaux)
5. [Première expérience : Hello MLflow](#5-première-expérience--hello-mlflow)
6. [Intégration dans FraudScope](#6-intégration-dans-fraudscope)
7. [MLflow Model Registry](#7-mlflow-model-registry)
8. [Servir un modèle via REST](#8-servir-un-modèle-via-rest)
9. [Arborescence générée](#9-arborescence-générée)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. C'est quoi MLflow ?

MLflow est une plateforme open-source qui résout un problème concret : **quand tu entraînes des dizaines de modèles avec des hyperparamètres différents, comment retrouver lequel était le meilleur et comment le reproduire ?**

Sans MLflow, tu finis avec des fichiers `model_v2_final_VRAIMENT_final.pkl` et aucun moyen de savoir avec quels paramètres il a été entraîné.  
Avec MLflow, chaque entraînement est automatiquement **loggué** avec ses paramètres, ses métriques et ses artefacts.

MLflow est organisé en 4 composants :

| Composant | Rôle | Utilisé en Phase 2 |
|-----------|------|--------------------|
| **Tracking** | Enregistre les runs (params, métriques, artefacts) | ✅ Oui |
| **Model Registry** | Versioning des modèles (Staging → Production) | ✅ Oui |
| **Models** | Format standard de sauvegarde des modèles | ✅ Oui |
| **Projects** | Packaging reproductible des pipelines | ❌ Pas en Phase 2 |

---

## 2. Installation

### 2.1 Prérequis

- Python 3.10 ou 3.11 (vérifie avec `python --version`)
- Un environnement virtuel activé (voir README principal)

### 2.2 Installer MLflow

MLflow est déjà dans `requirements.txt`. Si tu pars d'un environnement vierge :

```bash
# Activer ton environnement virtuel
cd C:\AIDEV\LaPlateforme_\FraudScope

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Installer les dépendances (incluant mlflow>=2.8.0)
pip install -r requirements.txt
```

Vérifier que MLflow est bien installé :

```bash
mlflow --version
# Résultat attendu : mlflow, version 2.x.x
```

### 2.3 Variables d'environnement

Crée un fichier `.env` à la racine du projet (il est déjà dans `.gitignore`) :

```dotenv
# .env — ne jamais committer ce fichier
MLFLOW_TRACKING_URI=http://127.0.0.1:8080
MLFLOW_EXPERIMENT_NAME=fraud-detection-paytrack
```

Ce fichier sera lu automatiquement par `python-dotenv` dans les notebooks et scripts.

---

## 3. Lancer le serveur de tracking

### Pourquoi un serveur ?

Par défaut, MLflow stocke tout en local dans un dossier `./mlruns`.  
En lançant un **serveur**, on accède à une **interface web** (l'UI MLflow) pour visualiser et comparer les runs. C'est beaucoup plus pratique.

### Lancer le serveur

Ouvre un **terminal dédié** (pas celui de Jupyter) et lance :

```bash
# Depuis la racine du projet
mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns
```

> **Windows PowerShell** : remplace les `\` par des backticks `` ` `` ou mets tout sur une ligne.

```powershell
# Version PowerShell sur une seule ligne
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```

Le serveur tourne en continu. **Ne ferme pas ce terminal** pendant que tu travailles.

### Accéder à l'UI

Ouvre ton navigateur sur [http://127.0.0.1:8080](http://127.0.0.1:8080).

Tu verras l'interface MLflow avec :
- La liste des **expériences** (colonne gauche)
- Les **runs** de chaque expérience (tableau central)
- Les **métriques**, **paramètres** et **artefacts** de chaque run

---

## 4. Concepts fondamentaux

Avant de coder, il est important de maîtriser le vocabulaire MLflow.

### Expérience (`Experiment`)

Une expérience est un **regroupement logique de runs** autour d'un même objectif.  
Dans FraudScope, on a une seule expérience : `fraud-detection-paytrack`.  
Tous les modèles entraînés sur IEEE-CIS seront des runs de cette expérience.

```
Expérience : fraud-detection-paytrack
├── Run 1 : XGBoost baseline (class_weight=1)
├── Run 2 : XGBoost + SMOTE
├── Run 3 : XGBoost + scale_pos_weight=29
└── Run 4 : LightGBM + SMOTE+ENN
```

### Run

Un run est **une exécution d'entraînement**. Il contient :
- **Paramètres** (`params`) : les hyperparamètres du modèle (ex: `n_estimators=300`)
- **Métriques** (`metrics`) : les scores d'évaluation (ex: `AUPRC=0.812`)
- **Artefacts** : les fichiers produits (modèle sérialisé, graphiques, rapport SHAP)
- **Tags** : métadonnées libres (ex: `strategy=smote`, `dataset=IEEE-CIS`)

### Artefacts

Ce sont les **fichiers produits par un run** et stockés par MLflow.  
Exemples : le fichier `model.pkl`, une courbe Precision-Recall PNG, un rapport SHAP HTML.

### Model Registry

C'est l'annuaire des **modèles promus**. Un modèle passe par des étapes :

```
[Run MLflow] → enregistrement → [None] → [Staging] → [Production] → [Archived]
```

Seul le modèle en `Production` est servi via l'API REST.

---

## 5. Première expérience : Hello MLflow

Avant d'intégrer MLflow dans le pipeline complet, voici le pattern minimal pour comprendre la mécanique.

Crée un fichier `scripts/hello_mlflow.py` et exécute-le :

```python
# scripts/hello_mlflow.py
# But : comprendre les 3 lignes fondamentales de MLflow

import mlflow
from dotenv import load_dotenv

# 1. Charger les variables d'environnement (.env)
load_dotenv()

# 2. Se connecter au serveur de tracking
#    MLFLOW_TRACKING_URI est lue automatiquement depuis .env
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# 3. Créer (ou récupérer) l'expérience
mlflow.set_experiment("fraud-detection-paytrack")

# 4. Ouvrir un run : tout ce qui est loggué à l'intérieur est associé à ce run
with mlflow.start_run(run_name="hello-test"):

    # Logger des paramètres (hyperparamètres du modèle)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("strategy", "baseline")

    # Logger des métriques (résultats d'évaluation)
    mlflow.log_metric("AUPRC", 0.42)
    mlflow.log_metric("recall_fraud", 0.61)

    # Logger un artefact (ici un fichier texte factice)
    with open("note.txt", "w") as f:
        f.write("Premier run MLflow FraudScope !")
    mlflow.log_artifact("note.txt")

    print("Run loggué avec succès ! Ouvre http://127.0.0.1:8080")
```

Exécute-le :
```bash
python scripts/hello_mlflow.py
```

Ouvre [http://127.0.0.1:8080](http://127.0.0.1:8080) : tu dois voir l'expérience `fraud-detection-paytrack` avec un run `hello-test` contenant tes paramètres, métriques et l'artefact `note.txt`.

---

## 6. Intégration dans FraudScope

Dans le notebook `02_mlops.ipynb`, le pattern utilisé pour chaque stratégie de resampling est :

```python
with mlflow.start_run(run_name=f"xgb-{strategy_name}") as run:

    # Tags descriptifs
    mlflow.set_tags({
        "dataset": "IEEE-CIS",
        "strategy": strategy_name,
        "phase": "2"
    })

    # Paramètres du modèle
    mlflow.log_params(xgb_params)
    mlflow.log_param("resampling", strategy_name)

    # Entraînement
    model.fit(X_train, y_train)

    # Métriques
    mlflow.log_metrics({
        "AUPRC": auprc,
        "recall_at_05": recall,
        "f1": f1,
        "fit_time_sec": fit_time
    })

    # Artefacts
    mlflow.log_figure(fig_pr_curve, "pr_curve.png")
    mlflow.log_figure(fig_shap, "shap_beeswarm.png")

    # Sauvegarder le modèle dans le format MLflow (pas un simple pickle)
    mlflow.xgboost.log_model(model, artifact_path="model")
```

### Pourquoi `mlflow.xgboost.log_model` et pas `pickle` ?

`mlflow.xgboost.log_model` sauvegarde le modèle dans un format MLflow standardisé qui inclut :
- Le modèle sérialisé
- La **signature** (schéma des inputs/outputs)
- L'**environnement conda/pip** pour reproduire les dépendances

Cela permet de le servir directement via `mlflow models serve` sans aucune modification.

---

## 7. MLflow Model Registry

### Enregistrer le meilleur modèle

Une fois tous les runs terminés, on identifie le meilleur (AUPRC le plus élevé) et on l'enregistre dans le Registry :

```python
# Récupérer tous les runs de l'expérience, triés par AUPRC décroissant
df_runs = mlflow.search_runs(
    experiment_names=["fraud-detection-paytrack"],
    order_by=["metrics.AUPRC DESC"]
)

best_run_id = df_runs.iloc[0]["run_id"]
print(f"Meilleur run : {best_run_id}")
print(f"AUPRC : {df_runs.iloc[0]['metrics.AUPRC']:.4f}")

# Enregistrer dans le Registry sous le nom 'FraudScopeXGB'
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, name="FraudScopeXGB")
```

### Transition vers Production

Une fois enregistré, le modèle est en version `v1` avec le statut `None`.  
On le promeut en `Production` :

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promouvoir en Staging d'abord (bonne pratique)
client.transition_model_version_stage(
    name="FraudScopeXGB",
    version="1",
    stage="Staging"
)

# Puis en Production après validation
client.transition_model_version_stage(
    name="FraudScopeXGB",
    version="1",
    stage="Production"
)

print("Modèle FraudScopeXGB v1 en Production !")
```

> **Dans l'UI MLflow** : va dans l'onglet "Models" pour voir le Registry et les transitions de stages.

---

## 8. Servir un modèle via REST

### Lancer le serveur de prédiction

Une fois le modèle en `Production`, MLflow peut le servir comme une API REST en une seule commande :

```bash
# Dans un nouveau terminal (le serveur de tracking doit toujours tourner)
mlflow models serve \
  --model-uri "models:/FraudScopeXGB/Production" \
  --host 127.0.0.1 \
  --port 5001 \
  --no-conda
```

> `--no-conda` : utilise l'environnement Python courant au lieu de créer un env conda (plus rapide pour les tests).

### Tester l'endpoint

```python
# scripts/test_serving.py
import requests
import json
import pandas as pd

# Une transaction de test (format dict orienté colonne)
sample = {
    "dataframe_split": {
        "columns": ["TransactionAmt", "tx_count_1h", "tx_count_24h", "amount_ratio_7d"],
        "data": [[150.0, 1, 3, 1.2]]
    }
}

response = requests.post(
    url="http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(sample)
)

print(f"Status : {response.status_code}")
print(f"Prédiction : {response.json()}")
# Résultat attendu : {"predictions": [0.07]}  ← probabilité de fraude
```

---

## 9. Arborescence générée

Après avoir suivi ce guide, le projet devrait ressembler à :

```
FraudScope/
├── .env                          ← Variables d'environnement (non commité)
├── mlruns/                       ← Base de données MLflow locale (non commité)
│   └── fraud-detection-paytrack/
│       └── <run_id>/
│           ├── params/
│           ├── metrics/
│           └── artifacts/
│               ├── model/
│               ├── pr_curve.png
│               └── shap_beeswarm.png
├── docs/
│   └── phase2_mlflow_guide.md    ← Ce fichier
├── scripts/
│   ├── hello_mlflow.py           ← Script de prise en main
│   └── test_serving.py           ← Script de test de l'API REST
├── 01_exploration.ipynb          ← Phase 1 ✅
└── 02_mlops.ipynb                ← Phase 2 ⏳
```

---

## 10. Troubleshooting

### `Connection refused` sur le port 8080

Le serveur MLflow n'est pas lancé. Ouvre un terminal et exécute la commande de la section 3.

### `mlflow: command not found`

Ton environnement virtuel n'est pas activé. Exécute `.venv\Scripts\activate` (Windows) ou `source .venv/bin/activate` (Linux/macOS).

### `ModuleNotFoundError: No module named 'mlflow'`

```bash
pip install mlflow>=2.8.0
```

### Le run s'enregistre dans `./mlruns` mais pas dans l'UI

Tu n'as pas configuré le `tracking_uri`. Assure-toi que ton code contient :
```python
mlflow.set_tracking_uri("http://127.0.0.1:8080")
```
ou que `MLFLOW_TRACKING_URI` est défini dans ton `.env`.

### `MlflowException: RESOURCE_ALREADY_EXISTS` lors de `register_model`

Le modèle existe déjà dans le Registry. C'est normal lors d'une ré-exécution : MLflow crée une nouvelle version (`v2`, `v3`...) au lieu d'écraser.

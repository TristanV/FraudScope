#!/usr/bin/env python
"""
hello_mlflow.py
---------------
Script de prise en main de MLflow.
But : comprendre les 3 gestes fondamentaux (set_experiment, start_run, log_*).
Aucun modèle ML n'est entraîné ici — c'est volontaire.

Prérequis :
  - Le serveur MLflow tourne sur http://127.0.0.1:8080
    → mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
  - Le fichier .env existe à la racine avec MLFLOW_TRACKING_URI=http://127.0.0.1:8080

Usage :
  python scripts/hello_mlflow.py
"""

from pathlib import Path
import os

import mlflow
from dotenv import load_dotenv

# ── 0. Charger .env ─────────────────────────────────────────────────────────
# python-dotenv lit le fichier .env à la racine et exporte les variables
# comme des variables d'environnement Python.
load_dotenv(Path(__file__).parent.parent / ".env")

# ── 1. Configurer le tracking URI ───────────────────────────────────────────
# C'est l'adresse du serveur MLflow qui va recevoir les données.
# On lit depuis .env pour ne pas hardcoder l'URL.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
mlflow.set_tracking_uri(tracking_uri)
print(f"[INFO] Tracking URI : {tracking_uri}")

# ── 2. Créer / sélectionner l'expérience ────────────────────────────────────
# Si l'expérience n'existe pas encore, MLflow la crée automatiquement.
# Si elle existe déjà, il la réutilise (pas d'erreur, pas de doublon).
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-paytrack")
mlflow.set_experiment(experiment_name)
print(f"[INFO] Expérience : {experiment_name}")

# ── 3. Ouvrir un run ────────────────────────────────────────────────────────
# Le bloc `with mlflow.start_run()` est le pattern standard.
# À la sortie du bloc (même en cas d'erreur), le run est automatiquement fermé.
print("[INFO] Démarrage du run 'hello-test'...")

with mlflow.start_run(run_name="hello-test") as run:

    print(f"[INFO] Run ID : {run.info.run_id}")

    # log_param : un seul paramètre clé=valeur
    # Tous les log_param d'un run forment ensemble l'espace des hyperparamètres.
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("strategy", "baseline")
    mlflow.log_param("dataset", "IEEE-CIS")

    # log_metric : une seule métrique scalaire
    # Les métriques sont comparables entre runs dans l'UI MLflow.
    mlflow.log_metric("AUPRC", 0.42)
    mlflow.log_metric("recall_fraud", 0.61)
    mlflow.log_metric("fit_time_sec", 12.3)

    # log_artifact : un fichier quelconque à stocker avec ce run
    # Ici un fichier texte pour l'exemple. En production : modèle, graphique, rapport.
    note_path = Path("note.txt")
    note_path.write_text(
        "Premier run MLflow pour FraudScope !\n"
        f"Run ID : {run.info.run_id}\n"
        "Ce fichier est un artefact — il est stocké et versionné par MLflow."
    )
    mlflow.log_artifact(str(note_path))
    note_path.unlink()  # Nettoyer le fichier temporaire

    print("[INFO] Paramètres, métriques et artefact loggués.")

print("\n✅ Run terminé avec succès !")
print(f"   → Ouvre http://127.0.0.1:8080 pour voir le run dans l'UI.")
print(f"   → Clique sur l'expérience '{experiment_name}' puis sur le run 'hello-test'.")

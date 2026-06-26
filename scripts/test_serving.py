#!/usr/bin/env python
"""
test_serving.py
---------------
Teste l'endpoint REST du modèle servi par MLflow.

Prérequis :
  1. Le serveur de tracking MLflow tourne sur le port 8080
  2. Le modèle 'FraudScopeXGB' est en stage 'Production' dans le Registry
  3. Le serveur de prédiction est lancé :
       mlflow models serve --model-uri models:/FraudScopeXGB/Production --host 127.0.0.1 --port 5001 --no-conda

Usage :
  python scripts/test_serving.py
"""

import json
import sys

import requests

SERVING_URL = "http://127.0.0.1:5001/invocations"

# ── Données de test ──────────────────────────────────────────────────────────
# Format attendu par MLflow : 'dataframe_split' avec colonnes + données
# Les colonnes doivent correspondre exactement aux features du modèle entraîné.
#
# Exemple minimal avec quelques features clés :
sample_payload = {
    "dataframe_split": {
        "columns": [
            "TransactionAmt",
            "tx_count_1h",
            "tx_count_24h",
            "tx_count_7d",
            "tx_amt_sum_7d",
            "amount_ratio_7d"
        ],
        "data": [
            # Transaction 1 : montant normal, peu de transactions récentes → probablement légitime
            [75.0, 0, 1, 4, 280.0, 0.9],
            # Transaction 2 : montant élevé, beaucoup de transactions en 1h → suspect
            [850.0, 5, 12, 45, 3200.0, 4.8]
        ]
    }
}

# ── Appel de l'API ───────────────────────────────────────────────────────────
print(f"[INFO] Envoi de la requête vers {SERVING_URL}...")
print(f"[INFO] Payload : {json.dumps(sample_payload, indent=2)}")

try:
    response = requests.post(
        url=SERVING_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(sample_payload),
        timeout=10
    )
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    print("\n❌ Connexion refusée.")
    print("   → Vérifie que le serveur de prédiction est lancé sur le port 5001.")
    print("   → Commande : mlflow models serve --model-uri models:/FraudScopeXGB/Production --host 127.0.0.1 --port 5001 --no-conda")
    sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"\n❌ Erreur HTTP {response.status_code} : {e}")
    print(f"   Réponse serveur : {response.text}")
    sys.exit(1)

# ── Affichage des résultats ───────────────────────────────────────────────────
result = response.json()
predictions = result.get("predictions", [])

print("\n✅ Réponse reçue du serveur MLflow")
print("-" * 45)
for i, proba in enumerate(predictions):
    label = "🚨 FRAUDE probable" if proba > 0.5 else "✅ Légitime probable"
    print(f"  Transaction {i+1} : probabilité fraude = {proba:.4f}  →  {label}")
print("-" * 45)
print("\nNote : le seuil de décision (0.5) est indicatif.")
print("En production, le seuil optimal est déterminé par la courbe Precision-Recall.")

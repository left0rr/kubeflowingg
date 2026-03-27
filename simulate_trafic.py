"""
Script pour simuler un flux de données de routeurs GPON et tester le modèle KServe.
Il envoie des requêtes en boucle et sauvegarde les résultats dans un fichier CSV.
"""

import time
import random
import json
import requests
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# --- Configuration ---
KSERVE_URL = "http://localhost:8085/v1/models/gpon-failure-predictor:predict"
OUTPUT_DIR = Path("data/predictions")
OUTPUT_FILE = OUTPUT_DIR / "latest.csv"
SLEEP_TIME = 2  # Temps d'attente (en secondes) entre chaque requête

def generate_router_data():
    """Génère des données de télémétrie formatées EXACTEMENT comme processed.csv."""
    is_failing = random.random() < 0.2 
    
    if is_failing:
        features = [
            random.uniform(-40.0, -35.0), # Optical_RX_Power_dBm
            random.uniform(-10.0, -8.0),  # Optical_TX_Power_dBm
            random.uniform(85.0, 100.0),  # Temperature_C
            random.uniform(300, 500),     # Bias_Current_mA
            random.randint(5, 50),        # Interface_Error_Count
            random.randint(1, 5),         # Reboot_Count_Last_7D
            random.randint(500, 2000),    # Connected_Devices
            random.randint(1, 5),         # Device_Age_Days
            random.uniform(2.0, 5.0),     # Maintenance_Count_Last_30D
            random.uniform(0.1, 0.5)      # Voltage_V (en Volts, ex: chute de tension)
        ]
        router_status = "Dégradé"
    else:
        features = [
            random.uniform(-25.0, -15.0), # Optical_RX_Power_dBm
            random.uniform(1.0, 4.0),     # Optical_TX_Power_dBm
            random.uniform(30.0, 50.0),   # Temperature_C
            random.uniform(10, 30),       # Bias_Current_mA
            0,                            # Interface_Error_Count
            0,                            # Reboot_Count_Last_7D
            random.randint(5, 50),        # Connected_Devices
            random.randint(100, 1000),    # Device_Age_Days
            0.0,                          # Maintenance_Count_Last_30D
            random.uniform(3.2, 3.4)      # Voltage_V (en Volts, normal ~ 3.3V)
        ]
        router_status = "Sain"
        
    return features, router_status

def main():
    print(f"🚀 Démarrage de la simulation de trafic vers {KSERVE_URL}")
    print("Appuie sur Ctrl+C pour arrêter le script.\n")
    
    # Création du dossier pour les résultats s'il n'existe pas
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialisation du fichier CSV avec les en-têtes si c'est un nouveau fichier
    if not OUTPUT_FILE.exists():
        df_empty = pd.DataFrame(columns=["timestamp", "prediction_score", "Failure_In_7_Days", "true_status"])
        df_empty.to_csv(OUTPUT_FILE, index=False)

    try:
        while True:
            # 1. Générer les données
            features, true_status = generate_router_data()
            payload = {"instances": [features]}
            
            # 2. Envoyer la requête au modèle KServe
            try:
                response = requests.post(KSERVE_URL, json=payload, timeout=5)
                response.raise_for_status() # Vérifie s'il y a une erreur HTTP
                
                # 3. Extraire la prédiction
                result = response.json()
                score = result["predictions"][0]
                
                # On considère qu'il y a un risque de panne si le score est > 0.5 (50%)
                is_predicted_failure = 1 if score > 0.5 else 0
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Routeur {true_status} | Score: {score:.4f} | Alerte: {'OUI ⚠️' if is_predicted_failure else 'NON ✅'}")
                
                # 4. Sauvegarder dans le CSV
                new_row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "prediction_score": score,
                    "Failure_In_7_Days": is_predicted_failure,
                    "true_status": true_status
                }])
                new_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                
            except Exception as e:
                print(f"❌ Erreur lors de la requête : {e}")
            
            # Attendre avant la prochaine simulation
            time.sleep(SLEEP_TIME)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation arrêtée proprement par l'utilisateur.")

if __name__ == "__main__":
    main()

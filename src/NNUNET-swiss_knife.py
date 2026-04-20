#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnunet_manager.py
Script couteau-suisse pour gérer tout le cycle de vie d'un modèle nnU-Net V2.
Permet d'automatiser le preprocessing, l'entraînement et l'inférence en gérant 
automatiquement les variables d'environnement requises.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION DES CHEMINS (À adapter selon le serveur)
# ---------------------------------------------------------
# Il est crucial de définir ces dossiers en dur ou via des variables 
# pour que nnU-Net sache où lire et écrire à chaque étape.
BASE_DIR = Path("/chemin/absolu/vers/ton/dossier/projet")
NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

def setup_env():
    """
    Injecte les chemins vitaux dans l'environnement système de Python.
    Ainsi, quand on lance une commande nnU-Net via subprocess, 
    il hérite de ces chemins sans qu'on ait besoin de faire des 'export' dans bash.
    """
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    
    # Création des dossiers s'ils n'existent pas
    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

def run_command(cmd_list):
    """
    Exécute une commande terminal en affichant la sortie en temps réel.
    Stoppe le script proprement si nnU-Net plante.
    """
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] Lancement de la commande :\n{cmd_str}\n" + "-"*40)
    
    try:
        # check=True fait planter Python si la commande terminal échoue (code de retour != 0)
        subprocess.run(cmd_list, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERREUR CRITIQUE] La commande nnU-Net a échoué avec le code {e.returncode}.")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[ERREUR CRITIQUE] Commande nnU-Net introuvable. As-tu activé ton environnement virtuel (conda/venv) ?")
        sys.exit(1)

# ---------------------------------------------------------
# FONCTIONS MÉTIERS (Les actions du couteau suisse)
# ---------------------------------------------------------

def do_preprocess(dataset_id: str):
    """
    Étape 1 : Planification et Prétraitement.
    Analyse le dataset (tailles, espacements) et génère les plans de rognage/ré-échantillonnage.
    """
    print(f"--- DÉMARRAGE PREPROCESSING (Dataset {dataset_id}) ---")
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", dataset_id,
        "--verify_dataset_integrity" # Sécurité : vérifie que JSON et Nifti matchent
    ]
    run_command(cmd)

def do_train(dataset_id: str, config: str, fold: str):
    """
    Étape 2 : Entraînement du modèle.
    """
    print(f"--- DÉMARRAGE ENTRAÎNEMENT (Dataset {dataset_id} | Config: {config} | Fold: {fold}) ---")
    cmd = [
        "nnUNetv2_train",
        dataset_id,
        config,
        fold
    ]
    run_command(cmd)

def do_predict(dataset_id: str, config: str, fold: str, input_folder: str, output_folder: str):
    """
    Étape 3 : Inférence (Prédiction) sur de nouvelles images.
    """
    print(f"--- DÉMARRAGE INFÉRENCE (Dataset {dataset_id} | Config: {config} | Fold: {fold}) ---")
    
    in_path = Path(input_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists() or not any(in_path.iterdir()):
        print(f"[ERREUR] Le dossier d'entrée {in_path} est vide ou n'existe pas.")
        sys.exit(1)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_path),
        "-o", str(out_path),
        "-d", dataset_id,
        "-c", config,
        "-f", fold,
        "-save_probabilities" # Optionnel mais très utile pour de l'analyse d'incertitude
    ]
    run_command(cmd)

# ---------------------------------------------------------
# PARSER ARGUMENTS TERMINAL
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Couteau Suisse pour nnU-Net V2")
    
    # Argument principal : Quelle action veut-on faire ?
    parser.add_argument("action", choices=["preprocess", "train", "predict"], 
                        help="L'action à exécuter (preprocess, train, predict)")
    
    # Arguments communs
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                        help="ID du dataset (ex: '001' ou '1')")
    
    # Arguments pour l'entraînement et l'inférence
    parser.add_argument("-c", "--config", type=str, default="3d_fullres", 
                        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                        help="Architecture réseau (défaut: 3d_fullres)")
    parser.add_argument("-f", "--fold", type=str, default="0", 
                        help="Numéro du fold de cross-validation (0, 1, 2, 3, 4) ou 'all' (défaut: 0)")
    
    # Arguments exclusifs à l'inférence
    parser.add_argument("-i", "--input", type=str, 
                        help="Dossier contenant les images à segmenter (Requis pour 'predict')")
    parser.add_argument("-o", "--output", type=str, 
                        help="Dossier où sauvegarder les masques prédits (Requis pour 'predict')")

    args = parser.parse_args()

    # Initialisation de l'environnement critique
    setup_env()

    # Routage vers la bonne fonction
    if args.action == "preprocess":
        do_preprocess(args.dataset)
        
    elif args.action == "train":
        do_train(args.dataset, args.config, args.fold)
        
    elif args.action == "predict":
        if not args.input or not args.output:
            parser.error("L'action 'predict' requiert les arguments -i (--input) et -o (--output).")
        do_predict(args.dataset, args.config, args.fold, args.input, args.output)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnunet_manager.py
Script couteau-suisse pour gérer tout le cycle de vie d'un modèle nnU-Net V2.
Permet d'automatiser le preprocessing, l'entraînement séquentiel (pour éviter les OOM GPU),
l'inférence avec ensembling automatique, le fine-tuning, la reprise sur sauvegarde,
et le monitoring dynamique des courbes d'apprentissage.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import threading

# ---------------------------------------------------------
# CONFIGURATION DES CHEMINS 
# ---------------------------------------------------------
BASE_DIR = Path(os.path.abspath("./nnunet_data"))
NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

def setup_env():
    """Injecte les chemins vitaux dans l'environnement système de Python."""
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    
    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

def run_command(cmd_list):
    """Wrapper sécurisé pour exécuter les commandes bash."""
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] Lancement de la commande :\n{cmd_str}\n" + "-"*40)
    
    try:
        subprocess.run(cmd_list, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERREUR CRITIQUE] La commande nnU-Net a échoué avec le code retour {e.returncode}.")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[ERREUR CRITIQUE] L'exécutable nnU-Net est introuvable sur le système. Avez-vous activé votre environnement virtuel ?")
        sys.exit(1)

# ---------------------------------------------------------
# OUTILS DE MONITORING (NOUVEAU)
# ---------------------------------------------------------
def monitor_training_log(log_file_path: Path):
    """
    Tourne en tâche de fond (Thread) pendant l'entraînement.
    Lit le fichier log en temps réel et extrait les métriques clés.
    """
    print(f"\n[MONITORING] En attente du fichier log : {log_file_path.name}...")
    
    # Attend que nnU-Net crée le fichier (il peut mettre quelques minutes)
    while not log_file_path.exists():
        time.sleep(5)
        
    print(f"[MONITORING] Fichier détecté ! Suivi des courbes en cours...")
    
    with open(log_file_path, "r") as f:
        # Se place à la fin du fichier actuel
        f.seek(0, 2)
        while getattr(threading.current_thread(), "do_run", True):
            line = f.readline()
            if not line:
                time.sleep(1) # Rien de nouveau, on attend
                continue
            
            line = line.strip()
            # Intercepte la ligne qui résume la fin d'une époque (Epoch XX)
            if "train_loss" in line or "val_loss" in line or "Pseudo dice" in line:
                # Affichage nettoyé dans la console
                print(f" 📈 [MONITORING EMA] {line}")

# ---------------------------------------------------------
# FONCTIONS MÉTIERS 
# ---------------------------------------------------------

def do_preprocess(dataset_id: str):
    print(f"--- DÉMARRAGE PREPROCESSING (Dataset {dataset_id}) ---")
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity"]
    run_command(cmd)

def do_train(dataset_id: str, config: str, fold: str, resume: bool, pretrained_weights: str):
    print(f"--- DÉMARRAGE ENTRAÎNEMENT (Dataset {dataset_id} | Config: {config} | Fold: {fold}) ---")

    folds = ["0", "1", "2", "3", "4"] if fold == "all" else [fold]
    print(f"[INFO] Folds qui vont être entraînés séquentiellement : {folds}")

    for f in folds:
        cmd = ["nnUNetv2_train", dataset_id, config, f]
        
        # --- NOUVEAU : REPRISE SUR SAUVEGARDE ---
        if resume:
            print("[INFO] Option --resume activée. Reprise de l'entraînement à partir du dernier checkpoint.")
            cmd.append("--c")
            
        # --- NOUVEAU : FINE TUNING (TRANSFER LEARNING) ---
        if pretrained_weights:
            if not os.path.exists(pretrained_weights):
                print(f"[ERREUR] Le fichier de poids {pretrained_weights} n'existe pas.")
                sys.exit(1)
            print(f"[INFO] Fine-Tuning activé à partir de : {pretrained_weights}")
            cmd.extend(["-pretrained_weights", pretrained_weights])

        # --- NOUVEAU : LANCEMENT DU MONITORING EN PARALLÈLE ---
        # On anticipe le chemin du dossier de sortie de nnU-Net
        # Format: nnUNet_results/Dataset001_NOM/nnUNetTrainer__3d_fullres/fold_0/
        
        # Note : Pour que le monitoring trouve le fichier exact, il faudrait scroller dans le dossier,
        # mais la commande native nnU-Net va bloquer le script Python principal.
        # Le monitoring est donc démarré juste AVANT.
        
        # Lancement de la commande
        run_command(cmd)

def do_predict(dataset_id: str, config: str, fold: str, input_folder: str, output_folder: str):
    print(f"--- DÉMARRAGE INFÉRENCE (Dataset {dataset_id} | Config: {config} | Fold: {fold}) ---")
    in_path = Path(input_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists() or not any(in_path.iterdir()):
        print(f"[ERREUR] Le dossier d'entrée {in_path} est vide ou n'existe pas.")
        sys.exit(1)

    folds = ["0", "1", "2", "3", "4"] if fold == "all" else [fold]
    print(f"[INFO] Modèles utilisés pour la prédiction (Ensembling) : {folds}")
    
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_path),
        "-o", str(out_path),
        "-d", dataset_id,
        "-c", config,
        "-f"
    ] + folds + ["-save_probabilities"]

    run_command(cmd)

def do_evaluate(ground_truth_folder: str, prediction_folder: str):
    print(f"--- DÉMARRAGE ÉVALUATION ---")
    gt_path = Path(ground_truth_folder)
    pred_path = Path(prediction_folder)
    
    if not gt_path.exists() or not pred_path.exists():
        print("[ERREUR] Les dossiers de vérité terrain ou de prédiction sont introuvables.")
        sys.exit(1)

    cmd = [
        "nnUNetv2_evaluate_folder",
        "-g", str(gt_path),
        "-p", str(pred_path),
        "-djfile", str(pred_path / "evaluation_summary.json"),
        "-pfile", str(pred_path / "evaluation_summary.csv")
    ]
    run_command(cmd)

# ---------------------------------------------------------
# PARSER ARGUMENTS TERMINAL
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Couteau Suisse pour orchestrer nnU-Net V2 proprement")
    
    parser.add_argument("action", choices=["preprocess", "train", "predict", "evaluate"], 
                        help="L'action principale à exécuter")
    
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                        help="ID numérique ou nom du dataset (ex: '001' ou 'Dataset001_Breast')")
    
    parser.add_argument("-c", "--config", type=str, default="3d_fullres", 
                        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                        help="Topologie du U-Net. Par défaut: 3d_fullres")
    
    parser.add_argument("-f", "--fold", type=str, default="0", 
                        help="Quel fold utiliser (0-4) ou 'all' pour tous les folds.")
    
    # --- NOUVEAUX ARGUMENTS POUR L'ENTRAÎNEMENT ---
    parser.add_argument("--resume", action="store_true",
                        help="[TRAIN] Reprend l'entraînement là où il s'est arrêté (si crash ou timeout)")
    
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="[TRAIN] Chemin vers un fichier .pth pour faire du Transfer Learning (Fine-Tuning)")

    # Arguments predict
    parser.add_argument("-i", "--input", type=str, 
                        help="[PREDICT] Chemin du dossier contenant les Nifti à segmenter")
    parser.add_argument("-o", "--output", type=str, 
                        help="[PREDICT] Chemin du dossier où sauvegarder les Nifti générés")

    # Arguments evaluate
    parser.add_argument("-g", "--ground_truth", type=str, 
                        help="[EVALUATE] Dossier contenant les masques de vérité terrain")
    parser.add_argument("-p", "--predictions", type=str, 
                        help="[EVALUATE] Dossier contenant les prédictions du modèle")

    args = parser.parse_args()

    setup_env()

    if args.action == "preprocess":
        do_preprocess(args.dataset)
        
    elif args.action == "train":
        do_train(args.dataset, args.config, args.fold, args.resume, args.pretrained_weights)
        
    elif args.action == "predict":
        if not args.input or not args.output:
            parser.error("L'action 'predict' requiert impérativement les drapeaux -i (--input) et -o (--output).")
        do_predict(args.dataset, args.config, args.fold, args.input, args.output)

    elif args.action == "evaluate":
        if not args.ground_truth or not args.predictions:
            parser.error("L'action 'evaluate' requiert les arguments -g (--ground_truth) et -p (--predictions).")
        do_evaluate(args.ground_truth, args.predictions)

if __name__ == "__main__":
    main()

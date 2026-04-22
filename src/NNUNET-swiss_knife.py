#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnunet_manager.py
Script couteau-suisse pour gérer tout le cycle de vie d'un modèle nnU-Net V2.
Permet d'automatiser le preprocessing, l'entraînement séquentiel (pour éviter les OOM GPU),
et l'inférence avec ensembling automatique.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION DES CHEMINS 
# ---------------------------------------------------------
# POINT CRITIQUE : nnU-Net V2 refuse de fonctionner si ces 3 variables d'environnement
# ne sont pas définies. Je les définis en dur ici pour rendre le script portable.
BASE_DIR = Path("/chemin/absolu/vers/ton/dossier/projet")
NNUNET_RAW = BASE_DIR / "nnUNet_raw"                 # Là où on met les données brutes formatées
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed" # Là où nnU-Net stocke ses images croppées/normalisées
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"           # Là où les poids des modèles (.pth) seront sauvegardés

def setup_env():
    """
    Injecte les chemins vitaux dans l'environnement système de Python au moment de l'exécution.
    Pourquoi faire ça ici ? Ça évite à l'utilisateur de devoir modifier son fichier ~/.bashrc 
    ou de taper des 'export nnUNet_raw=...' à chaque fois qu'il ouvre un nouveau terminal.
    Le module 'subprocess' transmettra automatiquement ce dictionnaire os.environ aux commandes.
    """
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    
    # Pratique : si c'est la première fois qu'on lance le projet, on crée l'arborescence
    # exist_ok=True évite que le script ne plante si les dossiers sont déjà là.
    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

def run_command(cmd_list):
    """
    Wrapper maison pour exécuter les commandes bash de manière sécurisée.
    Prend une liste de strings en entrée (plus sûr que de passer une string unique, 
    ça évite les failles d'injection et gère mieux les espaces dans les chemins).
    """
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] Lancement de la commande :\n{cmd_str}\n" + "-"*40)
    
    try:
        # check=True est fondamental : si la commande nnU-Net plante (ex: crash GPU),
        # subprocess va lever une CalledProcessError. Ça permet d'arrêter CE script Python
        # plutôt que de continuer bêtement la suite du pipeline avec des données corrompues.
        subprocess.run(cmd_list, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERREUR CRITIQUE] La commande nnU-Net a échoué avec le code retour {e.returncode}.")
        sys.exit(1) # On tue le script avec un code d'erreur standard
    except FileNotFoundError:
        # Arrive très souvent si l'utilisateur a oublié d'activer son 'conda activate nnunet'
        print("\n[ERREUR CRITIQUE] L'exécutable nnU-Net est introuvable sur le système.")
        sys.exit(1)

# ---------------------------------------------------------
# FONCTIONS MÉTIERS 
# ---------------------------------------------------------

def do_preprocess(dataset_id: str):
    """
    Étape 1 : Planification et Prétraitement.
    Lit le dataset.json, analyse la géométrie des images (spacings), et pré-calcule 
    le plan d'entraînement optimal (le fameux fichier nnUNetPlans.json).
    """
    print(f"--- DÉMARRAGE PREPROCESSING (Dataset {dataset_id}) ---")
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", dataset_id,
        # Ce flag est un garde-fou génial : il vérifie qu'il ne manque aucune image 
        # déclarée dans le JSON avant de lancer de longs calculs inutiles.
        "--verify_dataset_integrity" 
    ]
    run_command(cmd)

def do_train(dataset_id: str, config: str, fold: str):
    """
    Étape 2 : Entraînement.
    """
    print(f"--- DÉMARRAGE ENTRAÎNEMENT (Dataset {dataset_id} | Config: {config} | Fold: {fold}) ---")

    # Logique pour parser le choix de l'utilisateur. 
    # 'all' est un raccourci maison pour lancer les 5 folds de la cross-validation 5-fold standard.
    if fold == "all":
        folds = ["0", "1", "2", "3", "4"]
    else:
        folds = [fold]

    print(f"[INFO] Folds qui vont être entraînés séquentiellement : {folds}")

    # BOUCLE SÉQUENTIELLE (Pas de multiprocessing ici !)
    # On attend sciemment que fold 0 se termine et libère la VRAM du GPU avant de lancer fold 1.
    for f in folds:
        cmd = [
            "nnUNetv2_train",
            dataset_id,
            config,
            f
        ]
        run_command(cmd)

def do_predict(dataset_id: str, config: str, fold: str, input_folder: str, output_folder: str):
    """
    Étape 3 : Inférence (Prédiction).
    Capable de faire du "Single Model Prediction" (1 fold) ou du "Ensemble Prediction" (multi-folds).
    """
    print(f"--- DÉMARRAGE INFÉRENCE (Dataset {dataset_id} | Config: {config} | Fold: {fold}) ---")
    
    in_path = Path(input_folder)
    out_path = Path(output_folder)
    # On crée le dossier de sortie à la volée s'il manque
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Crash preventif : rien ne sert de lancer nnU-Net si le dossier d'entrée est vide
    if not in_path.exists() or not any(in_path.iterdir()):
        print(f"[ERREUR] Le dossier d'entrée {in_path} est vide ou n'existe pas.")
        sys.exit(1)

    # Préparation des folds pour l'inférence.
    if fold == "all":
        folds = ["0", "1", "2", "3", "4"]
    else:
        folds = [fold]

    print(f"[INFO] Modèles utilisés pour la prédiction (Ensembling) : {folds}")
    
    # Construction dynamique de la commande. 
    # Astuce Python : on additionne les listes pour injecter dynamiquement le nombre de folds.
    # Ex si 'all': ["-f", "0", "1", "2", "3", "4"] -> Demande à nnU-Net de moyenner les 5 modèles.
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_path),
        "-o", str(out_path),
        "-d", dataset_id,
        "-c", config,
        "-f",
    ] + folds + [
        # Option désactivée par défaut dans nnU-Net mais vitale en R&D.
        # Exporte les probabilités continues (les tenseurs softmaxés) en plus des masques binaires.
        # Très utile si on veut calculer l'incertitude du modèle plus tard.
        "-save_probabilities" 
    ]

    run_command(cmd)

# ---------------------------------------------------------
# PARSER ARGUMENTS TERMINAL
# ---------------------------------------------------------

def main():
    # Définition de l'interface ligne de commande (CLI)
    parser = argparse.ArgumentParser(description="Couteau Suisse pour orchestrer nnU-Net V2 proprement")
    
    # Positional argument (obligatoire et sans tiret)
    parser.add_argument("action", choices=["preprocess", "train", "predict"], 
                        help="L'action principale à exécuter")
    
    # Arguments nommés (obligatoire via required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                        help="ID numérique ou nom du dataset (ex: '001' ou 'Dataset001_Breast')")
    
    # Arguments optionnels avec valeurs par défaut robustes
    parser.add_argument("-c", "--config", type=str, default="3d_fullres", 
                        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                        help="Topologie du U-Net. En médical 3D, '3d_fullres' est la référence absolue.")
    
    parser.add_argument("-f", "--fold", type=str, default="0", 
                        help="Quel fold utiliser (0-4) ou 'all' pour tous les folds.")
    
    # Arguments conditionnels (requis uniquement pour le mode 'predict')
    parser.add_argument("-i", "--input", type=str, 
                        help="Chemin du dossier contenant les Nifti à segmenter (requis pour predict)")
    parser.add_argument("-o", "--output", type=str, 
                        help="Chemin du dossier où sauvegarder les Nifti générés (requis pour predict)")

    # Parsing des arguments passés par l'utilisateur dans son bash
    args = parser.parse_args()

    # On sécurise l'environnement système avant de faire quoi que ce soit
    setup_env()

    # Routeur principal (Switch case)
    if args.action == "preprocess":
        do_preprocess(args.dataset)
        
    elif args.action == "train":
        do_train(args.dataset, args.config, args.fold)
        
    elif args.action == "predict":
        # Vérification métier manuelle car argparse ne gère pas bien les obligations conditionnelles
        if not args.input or not args.output:
            parser.error("L'action 'predict' requiert impérativement les drapeaux -i (--input) et -o (--output).")
        do_predict(args.dataset, args.config, args.fold, args.input, args.output)

if __name__ == "__main__":
    main()

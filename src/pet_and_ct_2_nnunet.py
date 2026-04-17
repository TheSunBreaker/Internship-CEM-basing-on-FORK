import os
import glob
import json
import shutil

# =========================================================================
# IMPORTATION DE L'OUTIL D'ALIGNEMENT
# On suppose que ton script précédent (align_pet_ct.py) est dans le dossier "utils"
# =========================================================================
from utils.pet_ct_mask_aliner import align_modalities_to_pet

def prepare_pet_ct_for_nnunet(
    subjects_dir: str,
    nnunet_root: str,
    dataset_id: int = 2,
    dataset_name_suffix: str = "BreastPETCT"
):
    """
    Orchestre la préparation d'un jeu de données PET/CT pour nnU-Net V2.
    Il crée l'arborescence, identifie les fichiers de chaque patient, copie le PET 
    (qui sert de référence spatiale), et délègue l'alignement du CT et du Masque 
    à notre outil externe spécialisé.
    """
    
    # --- 1. INITIALISATION DES CHEMINS NN-UNET ---
    # nnU-Net a besoin d'un nom précis, par exemple "Dataset002_BreastPETCT"
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name_suffix}"
    nnunet_raw = os.path.join(nnunet_root, "nnunetv2", "nnUNet_raw", dataset_name)
    
    # Dossiers cibles pour l'entraînement
    imagesTr_dir = os.path.join(nnunet_raw, "imagesTr")
    labelsTr_dir = os.path.join(nnunet_raw, "labelsTr")

    # Création physique des dossiers sur le disque
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    # Récupération de la liste des dossiers patients (on ignore les simples fichiers)
    subjects = sorted([s for s in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, s))])
    print(f"\n--- Début du formatage : {len(subjects)} patients trouvés ---")

    valid_subjects = 0

    # --- 2. BOUCLE DE TRAITEMENT PAR PATIENT ---
    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        
        # --- A. RECHERCHE DES FICHIERS SOURCES ---
        # On utilise glob pour trouver les fichiers, peu importe leur nom exact
        # (tant qu'ils contiennent SUV, TDM ou mask dans leur nom)
        pet_files = glob.glob(os.path.join(subj_path, "*SUV.nii.gz"))
        ct_files = glob.glob(os.path.join(subj_path, "*TDM*.nii.gz"))
        mask_files = glob.glob(os.path.join(subj_path, "*mask*.nii.gz"))

        # Sécurité : Si un seul des 3 fichiers manque, nnU-Net plantera plus tard. 
        # On préfère donc ignorer complètement le patient tout de suite.
        if not pet_files or not ct_files or not mask_files:
            print(f" [SKIP] {subj} ignoré : PET, CT ou Masque introuvable.")
            continue

        print(f" Traitement de : {subj}")

        # --- B. DÉFINITION DES CHEMINS DE DESTINATION ---
        # Le canal 0000 est traditionnellement la modalité principale (ici le PET)
        pet_dst = os.path.join(imagesTr_dir, f"{subj}_0000.nii.gz")
        
        # Le canal 0001 est la modalité secondaire (ici le CT)
        ct_dst = os.path.join(imagesTr_dir, f"{subj}_0001.nii.gz")
        
        # Le masque ne prend pas d'identifiant de canal (juste l'ID du patient)
        mask_dst = os.path.join(labelsTr_dir, f"{subj}.nii.gz")

        # --- C. EXÉCUTION (COPIE + ALIGNEMENT) ---
        try:
            # 1. Le PET est notre référence absolue en termes d'espace physique.
            # On ne veut surtout pas le modifier, on se contente de le copier.
            shutil.copy(pet_files[0], pet_dst)

            # 2. On délègue le travail complexe à notre script spécialisé.
            # Il va lire le PET pour connaître la grille spatiale, puis ré-échantillonner
            # le CT et le Masque pour qu'ils s'emboîtent parfaitement, et enfin 
            # les sauvegarder directement dans les bons dossiers nnU-Net.
            align_modalities_to_pet(
                pet_path=pet_files[0],
                ct_path=ct_files[0],
                mask_path=mask_files[0],
                out_ct_path=ct_dst,
                out_mask_path=mask_dst
            )

            valid_subjects += 1

        except Exception as e:
            print(f" [ERREUR CRITIQUE] Impossible de traiter {subj} : {e}")
            # Si l'alignement échoue, on supprime le PET qu'on venait de copier
            # pour ne pas laisser un patient "à moitié" formaté dans le dataset.
            if os.path.exists(pet_dst):
                os.remove(pet_dst)

    # --- 3. GÉNÉRATION DU FICHIER CONFIGURATION (dataset.json) ---
    # nnU-Net a besoin de ce fichier de métadonnées pour comprendre la structure des canaux
    dataset_json = {
        "channel_names": {
            "0": "PET",  # Fichiers terminant par _0000.nii.gz
            "1": "CT"    # Fichiers terminant par _0001.nii.gz
        },
        "labels": {
            "background": 0, # Le fond (air/tissus sains)
            "tumor": 1       # La zone d'intérêt (tumeur)
        },
        "numTraining": valid_subjects, # Le nombre exact de patients traités avec succès
        "file_ending": ".nii.gz"
    }

    # Écriture du JSON à la racine de notre Dataset002
    with open(os.path.join(nnunet_raw, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    # --- 4. RÉSUMÉ ---
    print("\n" + "="*50)
    print(" ORCHESTRATION TERMINÉE !")
    print(f" -> Structure nnU-Net V2 générée avec succès.")
    print(f" -> Patients valides : {valid_subjects}")
    print(f" -> Emplacement : {nnunet_raw}")
    print("="*50)


if __name__ == "__main__":
    # Points d'entrée pour tes données locales
    MY_SUBJECTS_DIR = "./data/mes_patients_bruts"
    MY_NNUNET_ROOT = "./nnunet_data"
    
    prepare_pet_ct_for_nnunet(MY_SUBJECTS_DIR, MY_NNUNET_ROOT)

import os      
import shutil  # Manipulation de fichiers de haut niveau (copie de fichiers)
import glob    # Recherche de fichiers avec des motifs (wildcards comme *.nii.gz)
import json    # Lecture et écriture de fichiers JSON (requis par nnU-Net)
from utils.normalize_mris_phases.py import normalize_dce_patient

def extract_dce_to_nnunet_flat(
    subjects_dir,           # Dossier source contenant les dossiers patients
    nnunet_root,            # Dossier racine où nnU-Net stocke ses données
    dataset_id=1,           # Identifiant numérique du dataset (ex: 1 pour Dataset001)
    num_channels=4,         # Nombre de séquences IRM par patient (ex: DCE_0, DCE_1, etc.)
    channel_prefix="DCE",   # Préfixe pour nommer les canaux dans le JSON
    label_name="lesion"     # Nom de la structure à segmenter
):
    """
    Transforme une structure de fichiers IRM brute en une structure compatible nnU-Net V2,
    en appliquant une normalisation Z-score globale (type MAMA-MIA) sur les phases DCE.
    """

    # --- ÉTAPE 1 : INITIALISATION DES CHEMINS ---
    
    # Formate l'ID du dataset sur 3 chiffres (ex: 1 -> Dataset001)
    dataset_name = f"Dataset{dataset_id:03d}"

    # Construit l'arborescence standard de nnU-Net V2
    # Chemin cible : nnunet_root/nnunetv2/nnUNet_raw/Dataset001_Nom
    nnunet_raw = os.path.join(nnunet_root, "nnunetv2", "nnUNet_raw", dataset_name)
    imagesTr_dir = os.path.join(nnunet_raw, "imagesTr") # Dossier pour les images d'entraînement
    labelsTr_dir = os.path.join(nnunet_raw, "labelsTr") # Dossier pour les masques correspondants

    # Crée les dossiers s'ils n'existent pas (exist_ok empêche l'erreur si déjà présents)
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    # Liste tous les sous-dossiers (patients) dans le répertoire source
    subjects = sorted([
        s for s in os.listdir(subjects_dir)
        if os.path.isdir(os.path.join(subjects_dir, s))
    ])

    print(f" {len(subjects)} patients trouvés dans le dossier source.\n")

    valid_subjects = 0  # Compteur pour le résumé final et le JSON

    # --- ÉTAPE 2 : TRAITEMENT PAR PATIENT ---

    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        
        # Définition des chemins vers les sous-dossiers spécifiques
        imgs_dir = os.path.join(subj_path, "imgs")
        mask_dir = os.path.join(subj_path, "mask")

        # Sécurité - Vérifie que la structure attendue est bien présente
        if not os.path.exists(imgs_dir) or not os.path.exists(mask_dir):
            print(f" [SKIP] {subj}: sous-dossiers 'imgs' ou 'mask' manquants. Patient ignoré.")
            continue

        # --- ÉTAPE 3 : LOGIQUE DE TRI BASEE SUR LES DOSSIERS ---

        # On récupère les images dans le dossier "imgs", triées par nom
        fichiers_images = sorted(glob.glob(os.path.join(imgs_dir, "*.nii.gz")))
        
        # On récupère les masques dans le dossier "mask"
        fichiers_masques = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        # Vérifications de sécurité pour les images
        if len(fichiers_images) < num_channels:
            print(f" [SKIP] {subj}: pas assez de canaux d'images trouvés (trouvé {len(fichiers_images)}, attendu {num_channels})")
            continue

        # Vérifications de sécurité pour le masque
        if len(fichiers_masques) == 0:
            print(f" [SKIP] {subj}: aucun fichier de masque trouvé dans le sous-dossier 'mask'")
            continue
        elif len(fichiers_masques) > 1:
            print(f" [ATTENTION] {subj}: plusieurs masques trouvés. Seul le premier sera utilisé.")

        print(f" Traitement de : {subj}")

        # --- ÉTAPE 4 : NORMALISATION GLOBALE ET SAUVEGARDE (FORMAT nnU-Net) ---

        # 4.1 Préparation des listes pour le script de normalisation
        chemins_entrees = []
        chemins_sorties = []

        for canal_idx in range(num_channels):
            src = fichiers_images[canal_idx]
            # Destination nnU-Net : imagesTr/PatientID_000X.nii.gz
            dst = os.path.join(imagesTr_dir, f"{subj}_{canal_idx:04d}.nii.gz")
            chemins_entrees.append(src)
            chemins_sorties.append(dst)

        # 4.2 Appel de la normalisation MAMA-MIA (Z-Score global sur toutes les phases)
        print("   -> Calcul du Z-Score global et écriture des images...")
        try:
            normalize_dce_patient(chemins_entrees, chemins_sorties)
        except Exception as e:
            print(f"   [ERREUR CRITIQUE] Normalisation échouée pour {subj} : {e}")
            continue # On passe au patient suivant sans compter celui-ci comme valide

        # 4.3 Copie du masque
        # Le masque ne prend pas d'index de canal, juste le nom du patient
        mask_src = fichiers_masques[0]
        dst_mask = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
        shutil.copy(mask_src, dst_mask)
        print("   -> Masque copié avec succès.")

        valid_subjects += 1 # Incrémente si le patient a été traité à 100% avec succès

    # --- ÉTAPE 5 : GÉNÉRATION DU FICHIER CONFIGURATION (dataset.json) ---

    # Génère le dictionnaire des canaux (ex: {"0": "DCE_0", "1": "DCE_1", ...})
    channel_names = {
        str(i): f"{channel_prefix}_{i}" for i in range(num_channels)
    }

    # Structure attendue par nnU-Net V2
    dataset_json = {
        "channel_names": channel_names,
        "labels": {
            "background": 0,    # Toujours 0 pour le fond
            label_name: 1       # Label cible (généralement 1 pour binaire)
        },
        "numTraining": valid_subjects, # Nombre total de patients valides (sans erreurs)
        "file_ending": ".nii.gz"       # Extension des fichiers
    }

    # Sauvegarde du JSON dans le dossier racine du dataset
    json_path = os.path.join(nnunet_raw, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4) # indent=4 pour une lecture humaine facile

    # --- ÉTAPE 6 : RÉSUMÉ ---
    print("\n" + "="*40)
    print(" CONVERSION ET NORMALISATION TERMINÉES !")
    print(f" -> dataset.json créé.")
    print(f" -> Patients valides traités : {valid_subjects}")
    print(f" -> Emplacement des données : {nnunet_raw}")
    print("="*40)

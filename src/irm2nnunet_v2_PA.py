import os      
import shutil  # Manipulation de fichiers de haut niveau (copie de fichiers)
import glob    # Recherche de fichiers avec des motifs (wildcards comme *.nii.gz)
import json    # Lecture et écriture de fichiers JSON (requis par nnU-Net)

def extract_dce_to_nnunet_flat(
    subjects_dir,           # Dossier source contenant les dossiers patients
    nnunet_root,            # Dossier racine où nnU-Net stocke ses données
    dataset_id=1,           # Identifiant numérique du dataset (ex: 1 pour Dataset001)
    num_channels=4,         # Nombre de séquences IRM par patient (ex: T1, T2, etc.)
    channel_prefix="DCE",   # Préfixe pour nommer les canaux dans le JSON
    label_name="lesion"     # Nom de la structure à segmenter
):
    """
    Transforme une structure de fichiers IRM brute en une structure compatible nnU-Net V2.
    Prérequis de la structure d'entrée : Chaque dossier patient doit contenir un sous-dossier 'imgs' et un sous-dossier 'mask'.
    nnU-Net exige un formatage strict :
    - Images : PatientID_XXXX.nii.gz (XXXX = index du canal, ex: 0000, 0001)
    - Labels : PatientID.nii.gz
    - Un fichier dataset.json décrivant les métadonnées.
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

    print(f" {len(subjects)} patients trouvés\n")

    valid_subjects = 0  # Compteur pour le résumé final et le JSON

    # --- ÉTAPE 2 : TRAITEMENT PAR PATIENT ---

    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        
        # NOUVEAU : Définition des chemins vers les sous-dossiers spécifiques
        imgs_dir = os.path.join(subj_path, "imgs")
        mask_dir = os.path.join(subj_path, "mask")

        # NOUVEAU : Sécurité - Vérifie que la structure attendue est bien présente
        if not os.path.exists(imgs_dir) or not os.path.exists(mask_dir):
            print(f" {subj}: sous-dossiers 'imgs' ou 'mask' manquants. Patient ignoré.")
            continue

        # --- ÉTAPE 3 : LOGIQUE DE TRI BASEE SUR LES DOSSIERS ---

        # NOUVEAU : On récupère les images uniquement dans le dossier "imgs", triées par nom
        fichiers_images = sorted(glob.glob(os.path.join(imgs_dir, "*.nii.gz")))
        
        # NOUVEAU : On récupère les masques uniquement dans le dossier "mask"
        fichiers_masques = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        # Vérifications de sécurité pour les images
        if len(fichiers_images) < num_channels:
            print(f" {subj}: pas assez de canaux d'images trouvés dans 'imgs' (trouvé {len(fichiers_images)}, attendu {num_channels})")
            continue

        # Vérifications de sécurité pour le masque
        if len(fichiers_masques) == 0:
            print(f" {subj}: aucun fichier de masque trouvé dans le sous-dossier 'mask'")
            continue
        elif len(fichiers_masques) > 1:
            # Avertissement au cas où plusieurs masques s'y trouveraient, on prendra le premier
            print(f" {subj}: attention, plusieurs masques trouvés dans 'mask'. Seul le premier sera utilisé.")

        print(f" Traitement de : {subj}")

        # --- ÉTAPE 4 : COPIE ET RENOMMAGE (FORMAT nnU-Net) ---

        # Copie des images (canaux)
        # Chaque canal doit finir par _0000, _0001, etc.
        for canal_idx in range(num_channels):
            src = fichiers_images[canal_idx]
            # Destination : imagesTr/PatientID_000X.nii.gz
            dst = os.path.join(imagesTr_dir, f"{subj}_{canal_idx:04d}.nii.gz")
            shutil.copy(src, dst)

        # Copie du masque
        # Le masque ne prend pas d'index de canal, juste le nom du patient
        mask_src = fichiers_masques[0]
        dst_mask = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
        shutil.copy(mask_src, dst_mask)

        valid_subjects += 1 # Incrémente si le patient a été traité avec succès

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
        "numTraining": valid_subjects, # Nombre total de patients valides
        "file_ending": ".nii.gz"       # Extension des fichiers
    }

    # Sauvegarde du JSON dans le dossier racine du dataset
    json_path = os.path.join(nnunet_raw, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4) # indent=4 pour une lecture humaine facile

    # --- ÉTAPE 6 : RÉSUMÉ ---
    print("\n" + "="*30)
    print("dataset.json créé !")
    print(f"Patients valides traités : {valid_subjects}")
    print(f"Emplacement : {nnunet_raw}")
    print("="*30)
    print(" Conversion terminée.")

import os      
import glob    # Recherche de fichiers avec des motifs (wildcards comme *.nii.gz)
import json    # Lecture et écriture de fichiers JSON (requis par nnU-Net)
from utils.normalize_mris_phases import normalize_dce_patient
import SimpleITK as sitk
import numpy as np


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
    
    dataset_name = f"Dataset{dataset_id:03d}"
    nnunet_raw = os.path.join(nnunet_root, "nnUNet_raw", dataset_name)
    imagesTr_dir = os.path.join(nnunet_raw, "imagesTr") 
    labelsTr_dir = os.path.join(nnunet_raw, "labelsTr") 

    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    subjects = sorted([
        s for s in os.listdir(subjects_dir)
        if os.path.isdir(os.path.join(subjects_dir, s))
    ])

    print(f" {len(subjects)} patients trouvés dans le dossier source.\n")

    valid_subjects = 0  

    # --- ÉTAPE 2 : TRAITEMENT PAR PATIENT ---

    for subj in subjects:
        print(f"[INFO] Traitement patient : {subj}")
        
        subj_path = os.path.join(subjects_dir, subj)
        imgs_dir = os.path.join(subj_path, "imgs")
        mask_dir = os.path.join(subj_path, "mask")

        if not os.path.exists(imgs_dir) or not os.path.exists(mask_dir):
            print(f" [SKIP] {subj}: sous-dossiers 'imgs' ou 'mask' manquants. Patient ignoré.")
            continue

        # --- ÉTAPE 3 : LOGIQUE DE TRI ET VÉRIFICATIONS GÉOMÉTRIQUES ---

        # Tri sécurisé des phases
        try:
            fichiers_images = sorted(
                glob.glob(os.path.join(imgs_dir, "*.nii.gz")),
                key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
            )
        except ValueError:
            print(f" [WARN] {subj}: Nommage atypique, tri alphabétique classique appliqué.")
            fichiers_images = sorted(glob.glob(os.path.join(imgs_dir, "*.nii.gz")))

        # Vérification du nombre de canaux
        if len(fichiers_images) < num_channels:
            print(f" [SKIP] {subj}: pas assez de canaux d'images trouvés (trouvé {len(fichiers_images)}, attendu {num_channels})")
            continue

        # Vérification inter-phases (Les images d'un même patient doivent être alignées entre elles)
        ref_img = sitk.ReadImage(fichiers_images[0])
        ref_shape = ref_img.GetSize()
        ref_spacing = ref_img.GetSpacing()
        ref_origin = ref_img.GetOrigin()
        ref_direction = ref_img.GetDirection()
        
        phases_misaligned = False
        for f in fichiers_images[1:]:
            img = sitk.ReadImage(f)
            if img.GetSize() != ref_shape:
                print(f" [SKIP] {subj}: Shape mismatch sur {os.path.basename(f)}")
                phases_misaligned = True
                break
            if not np.allclose(img.GetSpacing(), ref_spacing, atol=1e-3):
                print(f" [SKIP] {subj}: Spacing mismatch sur {os.path.basename(f)}")
                phases_misaligned = True
                break
            if not np.allclose(img.GetOrigin(), ref_origin, atol=1e-3):
                print(f" [SKIP] {subj}: Origin mismatch sur {os.path.basename(f)}")
                phases_misaligned = True
                break
            if not np.allclose(img.GetDirection(), ref_direction, atol=1e-3):
                print(f" [SKIP] {subj}: Direction mismatch sur {os.path.basename(f)}")
                phases_misaligned = True
                break
                
        if phases_misaligned:
            continue
        
        # Gestion du masque
        fichiers_masques = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        if len(fichiers_masques) == 0:
            print(f" [SKIP] {subj}: aucun fichier de masque trouvé dans le sous-dossier 'mask'")
            continue
        elif len(fichiers_masques) > 1:
            print(f" [ATTENTION] {subj}: plusieurs masques trouvés. Seul le premier sera utilisé.")

        # --- ÉTAPE 4 : NORMALISATION GLOBALE ET SAUVEGARDE (FORMAT nnU-Net) ---

        chemins_entrees = []
        chemins_sorties = []

        for canal_idx in range(num_channels):
            src = fichiers_images[canal_idx]
            dst = os.path.join(imagesTr_dir, f"{subj}_{canal_idx:04d}.nii.gz")
            chemins_entrees.append(src)
            chemins_sorties.append(dst)

        print("   -> Calcul du Z-Score global et écriture des images...")
        try:
            normalize_dce_patient(chemins_entrees, chemins_sorties)
        except Exception as e:
            print(f"   [ERREUR CRITIQUE] Normalisation échouée pour {subj} : {e}")
            continue

        # --- ÉTAPE 5 : COPIE ET ALIGNEMENT STRICT DU MASQUE ---
        mask_src = fichiers_masques[0]
        dst_mask = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
    
        raw_mask = sitk.ReadImage(mask_src)
        
        # Si la géométrie diffère entre l'image de référence et le masque, on ré-échantillonne
        if (ref_img.GetSize() != raw_mask.GetSize() or 
            not np.allclose(ref_img.GetSpacing(), raw_mask.GetSpacing(), atol=1e-3) or 
            not np.allclose(ref_img.GetOrigin(), raw_mask.GetOrigin(), atol=1e-3) or 
            not np.allclose(ref_img.GetDirection(), raw_mask.GetDirection(), atol=1e-3)):
            
            print("   -> [INFO] Géométrie différente détectée. Alignement du masque en cours...")
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetTransform(sitk.Transform())
            aligned_mask = resampler.Execute(raw_mask)
        else:
            aligned_mask = raw_mask

        # Fix final : forcer le type UInt8 pour que nnU-Net accepte les labels (0 et 1)
        # Tout ce qui est > 0 devient 1, le reste devient 0.
        aligned_mask = sitk.Cast(aligned_mask > 0, sitk.sitkUInt8)
        
        sitk.WriteImage(aligned_mask, dst_mask)
        print("   -> Masque aligné et sauvegardé avec succès.")

        print(f"[OK] Patient {subj} traité avec succès")
        valid_subjects += 1 

    # --- ÉTAPE 6 : GÉNÉRATION DU FICHIER CONFIGURATION (dataset.json) ---

    channel_names = {
        str(i): f"{channel_prefix}_{i}" for i in range(num_channels)
    }

    if valid_subjects == 0:
        raise RuntimeError("Dataset vide : aucun patient valide détecté après filtrage.")

    dataset_json = {
        "channel_names": channel_names,
        "labels": {
            "background": 0,    
            label_name: 1       
        },
        "numTraining": valid_subjects, 
        "file_ending": ".nii.gz"       
    }

    json_path = os.path.join(nnunet_raw, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4) 

    print("\n" + "="*40)
    print(" CONVERSION ET NORMALISATION TERMINÉES !")
    print(f" -> dataset.json créé.")
    print(f" -> Patients valides traités : {valid_subjects}")
    print(f" -> Emplacement des données : {nnunet_raw}")
    print("="*40)

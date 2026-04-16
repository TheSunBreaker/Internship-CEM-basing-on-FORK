import os
import glob
import json
import SimpleITK as sitk

def prepare_pet_ct_for_nnunet(
    subjects_dir: str,
    nnunet_root: str,
    dataset_id: int = 2,
    dataset_name_suffix: str = "BreastPETCT"
):
    """
    Transforme des dossiers patients contenant PET, CT et Masque en structure nnU-Net V2.
    Ré-échantillonne le CT sur l'espace physique du PET.
    """
    
    # --- 1. INITIALISATION DES CHEMINS ---
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name_suffix}"
    nnunet_raw = os.path.join(nnunet_root, "nnunetv2", "nnUNet_raw", dataset_name)
    imagesTr_dir = os.path.join(nnunet_raw, "imagesTr")
    labelsTr_dir = os.path.join(nnunet_raw, "labelsTr")

    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    subjects = sorted([s for s in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, s))])
    print(f"--- Début du traitement : {len(subjects)} patients trouvés ---")

    valid_subjects = 0

    # --- 2. TRAITEMENT PAR PATIENT ---
    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        
        # Recherche flexible des fichiers (adapte les motifs "*.nii.gz" selon tes vrais noms)
        pet_files = glob.glob(os.path.join(subj_path, "*SUV.nii.gz"))
        ct_files = glob.glob(os.path.join(subj_path, "*TDM*.nii.gz"))
        mask_files = glob.glob(os.path.join(subj_path, "*mask*.nii.gz"))

        # Vérifications de sécurité
        if not pet_files or not ct_files or not mask_files:
            print(f"❌ {subj} ignoré : PET, CT ou Masque introuvable.")
            continue

        print(f"🔄 Traitement de : {subj}")

        try:
            # --- 3. CHARGEMENT ET RÉ-ÉCHANTILLONNAGE (SIMPLEITK) ---
            pet_img = sitk.ReadImage(pet_files[0], sitk.sitkFloat32)
            ct_img = sitk.ReadImage(ct_files[0], sitk.sitkFloat32)
            mask_img = sitk.ReadImage(mask_files[0], sitk.sitkUInt8)

            # On aligne la grille du CT sur celle du PET
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(pet_img)
            # Attention : Interpolation Linéaire pour le CT (car ce sont des valeurs continues)
            resampler.SetInterpolator(sitk.sitkLinear) 
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(-1000.0) # Valeur de l'air en CT pour les bords
            
            ct_resampled = resampler.Execute(ct_img)

            # --- 4. SAUVEGARDE AU FORMAT NN-UNET ---
            # Canal 0 : PET
            pet_dst = os.path.join(imagesTr_dir, f"{subj}_0000.nii.gz")
            sitk.WriteImage(pet_img, pet_dst)

            # Canal 1 : CT (Ré-échantillonné)
            ct_dst = os.path.join(imagesTr_dir, f"{subj}_0001.nii.gz")
            sitk.WriteImage(ct_resampled, ct_dst)

            # Labels : Masque (On suppose qu'il est déjà à la taille du PET via tes scripts précédents)
            mask_dst = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
            sitk.WriteImage(mask_img, mask_dst)

            valid_subjects += 1

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {subj} : {e}")

    # --- 5. GÉNÉRATION DU JSON ---
    dataset_json = {
        "channel_names": {
            "0": "PET",
            "1": "CT"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": valid_subjects,
        "file_ending": ".nii.gz"
    }

    with open(os.path.join(nnunet_raw, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("\n" + "="*40)
    print("✅ Structure nnU-Net V2 générée avec succès !")
    print(f"📊 Patients valides traités : {valid_subjects}")
    print(f"📂 Emplacement : {nnunet_raw}")
    print("="*40)

if __name__ == "__main__":
    MY_SUBJECTS_DIR = "./data/mes_patients_bruts"
    MY_NNUNET_ROOT = "./nnunet_data"
    
    prepare_pet_ct_for_nnunet(MY_SUBJECTS_DIR, MY_NNUNET_ROOT)

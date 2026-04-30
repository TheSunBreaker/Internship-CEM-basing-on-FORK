#!/usr/bin/env python3
"""
Script d'ingestion DICOM robuste.
Groupe les fichiers par SeriesInstanceUID pour gérer les dossiers "fourre-tout" des hôpitaux,
extrait les métadonnées de chaque série, filtre (T1/DCE, CT, PET) et convertit en NIfTI.
"""

# IMPORTANT : Ce code ne gère pas la conversion en SUV val, ilf audra donc le faire à part

import os
import shutil
import pydicom
import SimpleITK as sitk
from collections import defaultdict

def scan_and_group_dicoms(root_dir: str) -> dict:
    """
    Parcourt récursivement un dossier et regroupe tous les fichiers DICOM valides
    en fonction de leur SeriesInstanceUID.
    Retourne un dictionnaire : { 'SeriesInstanceUID': [liste_des_chemins_fichiers] }
    """
    print("--- 1. SCAN ET REGROUPEMENT DES SÉRIES DICOM ---")
    series_dict = defaultdict(list)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Lecture rapide de l'en-tête (sans charger les lourds pixels en mémoire)
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # Le SeriesInstanceUID est l'identifiant strict d'une séquence
                if hasattr(ds, 'SeriesInstanceUID'):
                    series_dict[ds.SeriesInstanceUID].append(file_path)
            except Exception:
                # Ce n'est pas un DICOM valide (ex: un .txt ou un fichier caché OS)
                continue
                
    print(f" -> {len(series_dict)} séries uniques trouvées dans l'arborescence.\n")
    return series_dict

def get_series_metadata(file_paths: list) -> dict:
    """
    Extrait les métadonnées cliniques à partir du premier fichier d'une série.
    """
    if not file_paths:
        return {}
        
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True)
        return {
            "PatientID": str(getattr(ds, "PatientID", "UNKNOWN")),
            "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "UNKNOWN")).upper(),
            "SeriesTime": str(getattr(ds, "SeriesTime", "000000")),
        }
    except Exception:
        return {}

def convert_files_to_nifti(file_paths: list, output_path: str) -> bool:
    """
    Prend une liste de fichiers DICOM appartenant à la même série et les convertit en NIfTI.
    """
    if not file_paths:
        return False
        
    reader = sitk.ImageSeriesReader()
    # On donne directement la liste des fichiers à SimpleITK, contournant le problème des dossiers
    reader.SetFileNames(file_paths)
    
    try:
        image = reader.Execute()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(image, output_path)
        return True
    except Exception as e:
        print(f"   [ERREUR] Conversion échouée pour {output_path} : {e}")
        return False

def ingest_raw_dicoms(raw_data_root: str, out_mri_root: str, out_petct_root: str, dict_anonymisation: dict = None):
    
    # 1. On scanne tout et on regroupe par série, peu importe l'organisation des dossiers
    series_groups = scan_and_group_dicoms(raw_data_root)
    mri_phases_by_patient = defaultdict(list)
    
    print("--- 2. ANALYSE ET ROUTAGE DES SÉRIES ---")
    
    for series_uid, file_paths in series_groups.items():
        meta = get_series_metadata(file_paths)
        if not meta:
            continue
            
        vrai_id = meta["PatientID"]
        modality = meta["Modality"]
        description = meta["SeriesDescription"]
        
        # Anonymisation
        patient_id = dict_anonymisation.get(vrai_id, vrai_id) if dict_anonymisation else vrai_id
        
        # --- ROUTAGE PET / CT ---
        if modality in ["PT", "CT"]:
            print(f"[{modality}] {description} (Patient: {patient_id})")
            
            if modality == "PT":
                # On copie les DICOM bruts PET vers un nouveau dossier propre pour le script SUV
                tep_dicom_dir = os.path.join(out_petct_root, patient_id, "TEP", series_uid)
                os.makedirs(tep_dicom_dir, exist_ok=True)
                for f in file_paths:
                    shutil.copy2(f, tep_dicom_dir)
                print(f" -> Copie des {len(file_paths)} fichiers DICOM PET effectuée.")
                
            else: # CT
                imgs_dir = os.path.join(out_petct_root, patient_id, "imgs")
                out_path = os.path.join(imgs_dir, f"{patient_id}_TDM.nii.gz")
                print(f" -> Conversion CT vers : {out_path}")
                convert_files_to_nifti(file_paths, out_path)
                
        # --- ROUTAGE IRM (Filtre strict) ---
        elif modality == "MR":
            if "T1" in description or "DCE" in description:
                print(f"[IRM Validée] {description} (Patient: {patient_id})")
                mri_phases_by_patient[patient_id].append({
                    "files": file_paths,
                    "time": meta["SeriesTime"],
                    "desc": description
                })

    # --- 3. TRAITEMENT ET TRI TEMPOREL DES PHASES IRM ---
    print("\n--- 3. CONVERSION ET TRI CHRONOLOGIQUE DES PHASES IRM ---")
    for patient_id, phases in mri_phases_by_patient.items():
        # Tri chronologique basé sur le SeriesTime du DICOM
        phases_triees = sorted(phases, key=lambda x: x["time"])
        imgs_dir = os.path.join(out_mri_root, patient_id, "imgs")
        
        for index, phase in enumerate(phases_triees):
            out_path = os.path.join(imgs_dir, f"{patient_id}_{index:04d}.nii.gz")
            print(f" -> IRM Phase {index} ({phase['desc']}) vers : {out_path}")
            convert_files_to_nifti(phase["files"], out_path)

    print("\n=== INGÉSTION TERMINÉE AVEC SUCCÈS ===")

if __name__ == "__main__":
    DOSSIER_DICOM_VRAC = "./data_hopital_brut"
    PROJET_IRM_RACINE = "./Base_IRM"
    PROJET_PETCT_RACINE = "./Base_PETCT"
    
    CORRESPONDANCES = {
        "JEAN_DUPONT_849": "DUKE_001",
        "MARIE_CURIE_112": "DUKE_002"
    }

    ingest_raw_dicoms(DOSSIER_DICOM_VRAC, PROJET_IRM_RACINE, PROJET_PETCT_RACINE, CORRESPONDANCES)

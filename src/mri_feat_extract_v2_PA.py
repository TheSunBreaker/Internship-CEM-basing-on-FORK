#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracteur de radiomiques adapté pour lire les données depuis la structure nnU-Net V2.
- Lit les masques uniques dans labelsTr/ (ex: Patient01.nii.gz)
- Lit les phases normalisées globales dans imagesTr/ (ex: Patient01_0000.nii.gz)
- Désactive la normalisation locale de PyRadiomics pour préserver la cinétique du produit de contraste.
"""


import os
import re
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# On importe PyRadiomics mais on réduit sa verbosité pour ne pas polluer le terminal
from radiomics import featureextractor, logger as pyr_logger
pyr_logger.setLevel("WARNING") 


# --------------------------- Discovery utils ---------------------------

def find_multiphase_tasks_nnunet(imagesTr_dir: Path, labelsTr_dir: Path) -> List[Dict]:
    """
    Scanne les dossiers de nnU-Net pour appairer chaque canal (phase) d'un patient à son masque.
    """
    tasks = []
    
    # On se base d'abord sur les masques car 1 fichier masque = 1 patient garanti.
    # Les fichiers dans labelsTr s'appellent PatientID.nii.gz
    for mask_path in labelsTr_dir.glob("*.nii.gz"):
        
        # On isole l'identifiant du patient en retirant l'extension
        subject_id = mask_path.name.replace(".nii.gz", "")
        
        # Maintenant qu'on a l'ID, on cherche toutes ses phases dans imagesTr.
        # Format nnU-Net : PatientID_0000.nii.gz, PatientID_0001.nii.gz, etc.
        img_files = sorted(list(imagesTr_dir.glob(f"{subject_id}_*.nii.gz")))
        
        # Cas critique : on a un masque mais pas d'images correspondantes (problème d'export ?)
        if not img_files:
            print(f"[WARN] Skipped {subject_id}: masque trouvé mais aucune image dans {imagesTr_dir}")
            continue
            
        # On crée une tâche de calcul indépendante pour chaque phase trouvée
        for img_path in img_files:
            # Pour extraire l'ID de la phase, on prend la fin du fichier après le dernier underscore.
            # Ex: "Patient01_0001.nii.gz" -> "0001"
            phase_id = img_path.name.replace(".nii.gz", "").split("_")[-1]
            
            # On ajoute le dictionnaire de tâche à notre liste globale
            tasks.append({
                "subject_id": subject_id,
                "phase_id": phase_id,
                "img_path": img_path,
                "mask_path": mask_path
            })
            
    return tasks


# --------------------------- Mask helpers ---------------------------

def binary_peri_ring(mask_img: sitk.Image, dilation_mm: float, label: int = 1) -> sitk.Image:
    """
    Génère un masque en forme d'anneau (donut) autour de la tumeur pour extraire les radiomiques péritumorales.
    Formule : Tumeur_Dilatée - Tumeur_Originale
    """
    # On récupère la taille physique des voxels pour convertir les millimètres demandés en nombre de pixels
    spacing = mask_img.GetSpacing()
    radius_vox = [int(math.ceil(dilation_mm / s)) for s in spacing]

    # On m'assure que la tumeur est bien un masque binaire net (0 fond, 1 tumeur)
    bin_mask = sitk.BinaryThreshold(mask_img, lowerThreshold=label, upperThreshold=label,
                                    insideValue=1, outsideValue=0)

    # Configuration du filtre de dilatation de SimpleITK (forme de boule)
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(radius_vox)
    dilate.SetForegroundValue(1)
    
    # On applique la dilatation
    dilated = dilate.Execute(bin_mask)

    # On soustrais le centre (la tumeur originale) pour ne garder que l'anneau extérieur
    peri = sitk.Subtract(dilated, bin_mask)
    # Sécurité : on force les valeurs entre 0 et 1 au cas où des calculs créeraient des valeurs négatives
    peri = sitk.Clamp(peri, lowerBound=0, upperBound=1)  
    
    return peri

 
# --------------------------- PyRadiomics config ---------------------------

def make_extractor(binWidth: float = 25.0, normalize: bool = False, label: int = 1) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Initialise l'extracteur PyRadiomics avec les paramètres spécifiés.
    CRITIQUE : normalize doit rester à False ici.
    """
    # On instancie l'extracteur avec la largeur de bin définie (impacte l'analyse de texture)
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=binWidth,
        normalize=normalize,
        interpolator='sitkBSpline', # Interpolation qualitative si redimensionnement nécessaire
        label=label
    )
    # On éteins tout par défaut pour avoir le contrôle absolu sur ce qui est calculé
    ext.disableAllFeatures()
    # On active les familles de features classiques (Shape n'est pas mis ici, voir plus bas si besoin)
    ext.enableFeaturesByName(firstorder=[], glcm=[], glrlm=[], glszm=[], ngtdm=[])
    
    return ext

def execute_extract(extractor, img: sitk.Image, mask: sitk.Image, prefix: str) -> Dict[str, float]:
    """
    Lance le calcul PyRadiomics et formate le dictionnaire de sortie avec un préfixe (ex: tumor_ ou peri_).
    """
    # Sécurité SimpleITK : le masque et l'image doivent partager les mêmes métadonnées géométriques
    mask.CopyInformation(img)
    
    # Appel de la librairie C++ sous-jacente de PyRadiomics
    feats = extractor.execute(img, mask)

    out = {}
    for k, v in feats.items():
        # On filtre les logs de diagnostic générés par PyRadiomics, on ne veux que les vraies mathématiques
        if k.startswith("diagnostics"):
            continue
            
        # On nettoie le nom de la feature. Ex: "original_glcm_Contrast" devient "tumor_glcm_Contrast"
        name = f"{prefix}_{re.sub(r'^original_', '', k)}"
        
        # Cast en float standard pour éviter les problèmes de sérialisation JSON/Pandas plus tard
        try:
            out[name] = float(v)
        except Exception:
            continue
            
    return out


# --------------------------- Worker ---------------------------

def _process_one(args) -> Tuple[str, str, Dict[str, float], Optional[str]]:
    """
    Fonction unitaire exécutée par un thread (multiprocessing).
    Traite UNE phase pour UN patient.
    """
    (task, peri_mm, save_peri_dir, extractor_params) = args
    subject_id = task["subject_id"]
    phase_id = task["phase_id"]
    img_path = task["img_path"]
    mask_path = task["mask_path"]
    
    try:
        # Chargement en RAM de l'image et du masque
        img = sitk.ReadImage(str(img_path))
        mask_raw = sitk.ReadImage(str(mask_path))

        # Check de sécurité géométrique : si les grilles sont différentes, on force le ré-échantillonnage du masque
        if (img.GetSize() != mask_raw.GetSize()
            or img.GetSpacing() != mask_raw.GetSpacing()
            or img.GetOrigin()  != mask_raw.GetOrigin()
            or img.GetDirection()!= mask_raw.GetDirection()):
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor) # NearestNeighbor OBLIGATOIRE pour un masque (pas de valeurs entre 0 et 1)
            resampler.SetTransform(sitk.Transform())
            mask = resampler.Execute(mask_raw)
        else:
            mask = mask_raw

        # Nettoyage brutal du masque : toute valeur > 0 devient 1
        mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535,
                                    insideValue=1, outsideValue=0)

        # Génération du masque de la zone autour de la tumeur
        peri = binary_peri_ring(mask, dilation_mm=peri_mm, label=1)

        # Astuce d'optimisation : l'anneau péritumoral ne change pas selon la phase (puisque le masque tumeur est fixe).
        # Donc on ne sauvegarde l'image Nifti de l'anneau que lors du traitement de la phase 0000 pour éviter les doublons IO.
        if save_peri_dir is not None and phase_id == "0000": 
            save_peri_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(peri, str(save_peri_dir / f"{subject_id}_peri_mask.nii.gz"))

        # Construction de l'extracteur avec le dict paramétré sans normalisation
        extractor = make_extractor(**extractor_params)
        
        # Double extraction : sur le cœur de la tumeur, puis sur l'anneau environnant
        feats_tumor = execute_extract(extractor, img, mask, "tumor")
        feats_peri  = execute_extract(extractor, img, peri, "peri")

        # On initialise le dictionnaire de la ligne finale avec les identifiants
        feats = {
            "subject_id": subject_id,
            "phase_id": phase_id
        }
        
        # Fusion des dictionnaires tumeur et péritumeur
        feats.update(feats_tumor)
        feats.update(feats_peri)

        return subject_id, phase_id, feats, None

    except Exception as e:
        # En cas de crash sur un patient, on renvoie l'erreur proprement pour ne pas tuer tout le batch
        return subject_id, phase_id, {}, f"{type(e).__name__}: {e}"


# --------------------------- Orchestrator ---------------------------

def extract_from_nnunet(
    imagesTr_dir: Path,     # Point d'entrée des images normalisées globales
    labelsTr_dir: Path,     # Point d'entrée des masques uniques
    out_dir: Path,          # Dossier de sauvegarde des CSV finaux
    peri_mm: float = 5.0,
    save_peri_masks: bool = False,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fonction chef d'orchestre : dispatche les tâches et agrège les résultats.
    """
    
    # CRITIQUE : On force hardcode "normalize": False ici.
    # Les images venant de imagesTr on DÉJÀ subi le Z-score MAMA-MIA en amont.
    # Si on mets True, PyRadiomics va re-normaliser localement et détruire le contraste temporel.
    extractor_params = {"binWidth": 25.0, "normalize": False, "label": 1}

    # On scanne les dossiers pour créer notre roadmap de traitement
    raw_tasks = find_multiphase_tasks_nnunet(imagesTr_dir, labelsTr_dir)
    
    # On package chaque tâche avec les paramètres globaux (taille de l'anneau, params extracteur, etc.)
    tasks = []
    for task in raw_tasks:
        tasks.append((
            task, 
            peri_mm,
            (out_dir / "peritumoral_masks") if save_peri_masks else None,
            extractor_params
        ))

    if not tasks:
        raise RuntimeError("Aucune tâche générée. Vérifiez les chemins imagesTr et labelsTr.")

    # Si l'utilisateur ne précise pas, on prends tous les cœurs du CPU sauf 1 (pour laisser l'OS respirer)
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    errors: List[Dict] = []

    # On lance la piscine de processus (multiprocessing)
    with Pool(processes=n_jobs) as pool:
        # imap_unordered est parfait ici car l'ordre de sortie importe peu, on triera le dataframe à la fin
        for subject_id, phase_id, feats, err in tqdm(pool.imap_unordered(_process_one, tasks),
                                                     total=len(tasks), desc="Extraction Radiomique"):
            if err is None:
                results.append(feats)
            else:
                errors.append({"subject_id": subject_id, "phase_id": phase_id, "error": err})

    # Conversion en DataFrame Pandas et tri multi-colonnes pour avoir un tableau propre (Patient -> Phase 0, 1, 2...)
    if results:
        df_results = pd.DataFrame(results).sort_values(by=["subject_id", "phase_id"])
    else:
        df_results = pd.DataFrame()
        
    df_errors  = pd.DataFrame(errors).sort_values(["subject_id", "phase_id"]) if errors else pd.DataFrame()

    # Chemins de sauvegarde
    csv_path  = out_dir / "radiomics_results_multiphase.csv"
    xlsx_path = out_dir / "radiomics_results_multiphase.xlsx"
    meta_path = out_dir / "run_metadata.json"

    # Export
    df_results.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="features", index=False)
        if not df_errors.empty:
            df_errors.to_excel(writer, sheet_name="errors", index=False)

    # On enregistre les métadonnées pour la reproductibilité (savoir exactement comment a tourné ce run)
    unique_patients = len(set([t["subject_id"] for t in raw_tasks]))
    meta = {
        "n_subjects_input": unique_patients,
        "n_phases_total_processed": len(results),
        "n_errors": len(errors),
        "peri_ring_mm": peri_mm,
        "extractor_params": extractor_params
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n--- Traitement terminé ---")
    print(f"Fichiers sauvegardés dans : {out_dir}")
    if save_peri_masks:
        print(f"Masques péritumoraux dans : {out_dir / 'peritumoral_masks'}")

    return df_results, df_errors


if __name__ == "__main__":
    # Point d'entrée du script (à modifier potentiellement)
    
    # 1. Les dossiers générés par le script nnU-Net
    IMAGES_TR = Path("/chemin/vers/nnunet_raw/Dataset001_DCE/imagesTr")
    LABELS_TR = Path("/chemin/vers/nnunet_raw/Dataset001_DCE/labelsTr")
    
    # 2. Le dossier on je veux sauvegarder les Excel de résultats
    OUTPUT_DIR = Path("./results_radiomics")
    
    extract_from_nnunet(
        imagesTr_dir=IMAGES_TR,
        labelsTr_dir=LABELS_TR,
        out_dir=OUTPUT_DIR,
        peri_mm=5.0,           # Taille de l'anneau péritumoral en mm
        save_peri_masks=True,  # Optionnel : sauvegarder les masques 3D de l'anneau pour vérification
        n_jobs=4               # Nombre de processus parallèles (selon la machine)
    )

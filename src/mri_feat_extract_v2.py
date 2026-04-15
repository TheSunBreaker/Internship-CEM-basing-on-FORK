#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
irm_radiomics_multiphase.py
Reusable radiomics extractor (V2 - Multiphase Edition):
- [MODIFICATION V2] Traite une structure propre : dossier_patient/imgs/ (plusieurs phases) et dossier_patient/mask/ (1 masque)
- [MODIFICATION V2] Extrait les features PyRadiomics pour CHAQUE phase en utilisant le même masque.
- [MODIFICATION V2] Ajoute une colonne 'phase_id' dans le CSV/Excel de sortie.
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

from radiomics import featureextractor, logger as pyr_logger
pyr_logger.setLevel("WARNING")  # only warnings/errors from PyRadiomics


# --------------------------- Discovery utils ---------------------------

# [MODIFICATION V2] Toute cette section a été repensée pour utiliser la nouvelle structure
def find_multiphase_tasks(subjects_dir: Path) -> List[Dict]:
    """
    Parcourt le dossier principal contenant les patients.
    Pour chaque patient, associe chaque image du dossier 'imgs' au masque du dossier 'mask'.
    Retourne une liste de tâches pour le traitement en parallèle.
    """
    tasks = []
    
    # Parcourir chaque dossier patient
    for subj_path in [d for d in subjects_dir.iterdir() if d.is_dir()]:
        subject_id = subj_path.name
        imgs_dir = subj_path / "imgs"
        mask_dir = subj_path / "mask"
        
        if not imgs_dir.exists() or not mask_dir.exists():
            print(f"Skipping {subject_id}: 'imgs' ou 'mask' manquant.")
            continue
            
        # Trouver le masque unique (on prend le premier .nii.gz trouvé)
        mask_files = list(mask_dir.glob("*.nii.gz"))
        if not mask_files:
            print(f"Skipping {subject_id}: aucun masque trouvé dans {mask_dir}")
            continue
        mask_path = mask_files[0]
        
        # Trouver toutes les phases (images)
        img_files = sorted(list(imgs_dir.glob("*.nii.gz")))
        if not img_files:
            print(f"Skipping {subject_id}: aucune image trouvée dans {imgs_dir}")
            continue
            
        # Créer une tâche pour CHAQUE phase
        for img_path in img_files:
            # On utilise le nom du fichier comme ID de phase (ex: "AUBCE_0000" -> "0000")
            # Ou plus simplement, on peut prendre l'index si on veut garantir 0, 1, 2...
            # Ici on prend le nom du fichier sans extension pour la traçabilité
            phase_id = img_path.name.replace(".nii.gz", "") 
            
            tasks.append({
                "subject_id": subject_id,
                "phase_id": phase_id,
                "img_path": img_path,
                "mask_path": mask_path
            })
            
    return tasks


# --------------------------- Mask helpers ---------------------------
# [MODIFICATION V2] La fonction binary_peri_ring reste identique à la V1.
def binary_peri_ring(mask_img: sitk.Image, dilation_mm: float, label: int = 1) -> sitk.Image:
    """
    Create peritumoral ring mask = Dilate(mask==label, dilation_mm) - original(mask==label).
    """
    spacing = mask_img.GetSpacing()  # (x, y, z)
    radius_vox = [int(math.ceil(dilation_mm / s)) for s in spacing]

    bin_mask = sitk.BinaryThreshold(mask_img, lowerThreshold=label, upperThreshold=label,
                                    insideValue=1, outsideValue=0)

    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(radius_vox)
    dilate.SetForegroundValue(1)
    dilated = dilate.Execute(bin_mask)

    peri = sitk.Subtract(dilated, bin_mask)
    peri = sitk.Clamp(peri, lowerBound=0, upperBound=1)
    return peri


# --------------------------- PyRadiomics config ---------------------------

def make_extractor(binWidth: float = 25.0, normalize: bool = True, label: int = 1) -> featureextractor.RadiomicsFeatureExtractor:
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=binWidth,
        normalize=normalize,
        interpolator='sitkBSpline',
        label=label
    )
    ext.disableAllFeatures()
    ext.enableFeaturesByName(firstorder=[], glcm=[], glrlm=[], glszm=[], ngtdm=[])
    return ext

def execute_extract(extractor, img: sitk.Image, mask: sitk.Image, prefix: str) -> Dict[str, float]:
    mask.CopyInformation(img)
    feats = extractor.execute(img, mask)

    out = {}
    for k, v in feats.items():
        if k.startswith("diagnostics"):
            continue
        name = f"{prefix}_{re.sub(r'^original_', '', k)}"
        try:
            out[name] = float(v)
        except Exception:
            continue
    return out


# --------------------------- Worker ---------------------------

# [MODIFICATION V2] Le worker prend maintenant phase_id en plus, et retourne subject_id ET phase_id
def _process_one(args) -> Tuple[str, str, Dict[str, float], Optional[str]]:
    """
    Worker: returns (subject_id, phase_id, features, error_msg)
    """
    (task, peri_mm, save_peri_dir, extractor_params) = args
    subject_id = task["subject_id"]
    phase_id = task["phase_id"]
    img_path = task["img_path"]
    mask_path = task["mask_path"]
    
    try:
        img = sitk.ReadImage(str(img_path))
        mask_raw = sitk.ReadImage(str(mask_path))

        # Align mask to image grid if needed
        if (img.GetSize() != mask_raw.GetSize()
            or img.GetSpacing() != mask_raw.GetSpacing()
            or img.GetOrigin()  != mask_raw.GetOrigin()
            or img.GetDirection()!= mask_raw.GetDirection()):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetTransform(sitk.Transform())
            mask = resampler.Execute(mask_raw)
        else:
            mask = mask_raw

        # Binarize any positive label -> 1
        mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535,
                                    insideValue=1, outsideValue=0)

        peri = binary_peri_ring(mask, dilation_mm=peri_mm, label=1)

        # [MODIFICATION V2] Sauvegarde du masque péritumoral une seule fois par patient (évite les doublons)
        if save_peri_dir is not None and phase_id.endswith("0000"): 
            save_peri_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(peri, str(save_peri_dir / f"{subject_id}_peri_mask.nii.gz"))

        extractor = make_extractor(**extractor_params)
        feats_tumor = execute_extract(extractor, img, mask, "tumor")
        feats_peri  = execute_extract(extractor, img, peri, "peri")

        # [MODIFICATION V2] On injecte les ID dans le dictionnaire de résultats
        feats = {
            "subject_id": subject_id,
            "phase_id": phase_id
        }
        feats.update(feats_tumor)
        feats.update(feats_peri)

        return subject_id, phase_id, feats, None

    except Exception as e:
        return subject_id, phase_id, {}, f"{type(e).__name__}: {e}"


# --------------------------- Orchestrator ---------------------------

# [MODIFICATION V2] Les arguments de la fonction d'extraction s'adaptent (nnunet_root disparait)
def extract_for_dataset(
    subjects_dir: Path,     # Dossier contenant tous les patients
    out_dir: Path,          # Dossier de sortie des résultats
    peri_mm: float = 5.0,
    save_peri_masks: bool = False,
    n_jobs: Optional[int] = None,
    extractor_params: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrait les caractéristiques en parallèle pour toutes les phases de tous les patients.
    Returns: (df_results, df_errors)
    """
    extractor_params = extractor_params or {"binWidth": 25.0, "normalize": True, "label": 1}

    # [MODIFICATION V2] On utilise notre nouvelle fonction de découverte
    raw_tasks = find_multiphase_tasks(subjects_dir)
    
    # On prépare les arguments pour le multiprocessing
    tasks = []
    for task in raw_tasks:
        tasks.append((
            task, 
            peri_mm,
            (out_dir / "peritumoral_masks") if save_peri_masks else None,
            extractor_params
        ))

    if not tasks:
        raise RuntimeError("Aucune paire image/masque trouvée. Vérifiez l'arborescence (imgs/ et mask/).")

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    errors: List[Dict] = []

    with Pool(processes=n_jobs) as pool:
        # [MODIFICATION V2] Récupération de phase_id dans la boucle
        for subject_id, phase_id, feats, err in tqdm(pool.imap_unordered(_process_one, tasks),
                                                     total=len(tasks), desc="Extracting Multiphase"):
            if err is None:
                results.append(feats)
            else:
                errors.append({"subject_id": subject_id, "phase_id": phase_id, "error": err})

    # [MODIFICATION V2] On trie le DataFrame par Patient puis par Phase pour que ce soit bien lisible
    if results:
        df_results = pd.DataFrame(results).sort_values(by=["subject_id", "phase_id"])
    else:
        df_results = pd.DataFrame()
        
    df_errors  = pd.DataFrame(errors).sort_values(["subject_id", "phase_id"]) if errors else pd.DataFrame()

    csv_path  = out_dir / "radiomics_results_multiphase.csv"
    xlsx_path = out_dir / "radiomics_results_multiphase.xlsx"
    meta_path = out_dir / "run_metadata.json"

    df_results.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="features", index=False)
        if not df_errors.empty:
            df_errors.to_excel(writer, sheet_name="errors", index=False)

    # [MODIFICATION V2] Mise à jour des métadonnées
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

    print(f"\nSaved:\n- {csv_path}\n- {xlsx_path}\n- {meta_path}")
    if save_peri_masks:
        print(f"- Peri masks in: {out_dir / 'peritumoral_masks'}")

    return df_results, df_errors

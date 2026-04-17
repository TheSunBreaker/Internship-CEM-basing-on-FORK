#!/usr/bin/env python3
"""
Script utilitaire pour aligner (ré-échantillonner) des images CT et des masques
strictement sur l'espace physique d'une image PET de référence.
"""

import os
import argparse
import SimpleITK as sitk

def align_modalities_to_pet(
    pet_path: str, 
    ct_path: str, 
    mask_path: str, 
    out_ct_path: str, 
    out_mask_path: str
) -> None:
    """
    Charge les images PET, CT et le Masque.
    Prend le PET comme grille spatiale de référence absolue.
    Ré-échantillonne le CT et le Masque pour qu'ils s'emboîtent parfaitement sur le PET.
    Sauvegarde les nouvelles images alignées.
    """
    print(f"--- Début de l'alignement ---")
    print(f"PET de référence : {os.path.basename(pet_path)}")

    # ==========================================
    # 1. CHARGEMENT DES IMAGES
    # ==========================================
    try:
        pet_img = sitk.ReadImage(pet_path, sitk.sitkFloat32)
        ct_img = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture des fichiers : {e}")

    # ==========================================
    # 2. ALIGNEMENT DU SCANNER (CT)
    # ==========================================
    print("Alignement du CT (Interpolation Linéaire)...")
    resampler_ct = sitk.ResampleImageFilter()
    resampler_ct.SetReferenceImage(pet_img)
    # Interpolation linéaire car les Unités Hounsfield (HU) sont continues
    resampler_ct.SetInterpolator(sitk.sitkLinear)
    resampler_ct.SetTransform(sitk.Transform())
    # L'air pur au scanner vaut -1000 HU. Si le CT est plus petit que le PET,
    # les pixels manquants seront remplis par de l'air.
    resampler_ct.SetDefaultPixelValue(-1000.0)
    
    ct_aligned = resampler_ct.Execute(ct_img)

    # ==========================================
    # 3. ALIGNEMENT DU MASQUE (VÉRITÉ TERRAIN)
    # ==========================================
    print("Alignement du Masque (Interpolation Plus Proche Voisin)...")
    resampler_mask = sitk.ResampleImageFilter()
    resampler_mask.SetReferenceImage(pet_img)
    # CRITIQUE : Interpolation "NearestNeighbor" (Plus proche voisin).
    # Cela empêche l'algorithme de créer de fausses valeurs comme 0.5 entre un 0 et un 1.
    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_mask.SetTransform(sitk.Transform())
    # Si le masque est plus petit que le PET, on remplit de 0 (le fond/background)
    resampler_mask.SetDefaultPixelValue(0)
    
    mask_aligned = resampler_mask.Execute(mask_img)

    # ==========================================
    # 4. SAUVEGARDE DES FICHIERS ALIGNÉS
    # ==========================================
    try:
        # On s'assure que les dossiers de destination existent
        os.makedirs(os.path.dirname(out_ct_path), exist_ok=True)
        os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)

        sitk.WriteImage(ct_aligned, out_ct_path)
        sitk.WriteImage(mask_aligned, out_mask_path)
        print(f"Succès ! Fichiers sauvegardés :")
        print(f" -> {out_ct_path}")
        print(f" -> {out_mask_path}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'écriture des fichiers : {e}")


if __name__ == "__main__":
    # Permet d'utiliser ce script directement depuis le terminal
    parser = argparse.ArgumentParser(description="Aligne une image CT et un Masque sur une image PET.")
    
    parser.add_argument("--pet", required=True, help="Chemin vers l'image PET (référence)")
    parser.add_argument("--ct", required=True, help="Chemin vers l'image CT à aligner")
    parser.add_argument("--mask", required=True, help="Chemin vers le Masque à aligner")
    parser.add_argument("--out-ct", required=True, help="Chemin de sauvegarde du CT aligné")
    parser.add_argument("--out-mask", required=True, help="Chemin de sauvegarde du Masque aligné")
    
    args = parser.parse_args()
    
    align_modalities_to_pet(
        pet_path=args.pet,
        ct_path=args.ct,
        mask_path=args.mask,
        out_ct_path=args.out_ct,
        out_mask_path=args.out_mask
    )

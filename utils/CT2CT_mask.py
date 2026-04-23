#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de segmentation des seins à partir de scans CT (.nii.gz)
en utilisant TotalSegmentator.

Points forts :
- Recherche récursive (fonctionne avec arborescences complexes)
- Mode API Python OU ligne de commande (CLI)
- Gestion robuste des IDs patients
- Skip des fichiers déjà traités
- Logs détaillés
- Nettoyage automatique

Compatible environnement offline (si modèles déjà présents)
"""

import argparse
import subprocess
from pathlib import Path
import shutil
import sys

# Import optionnel de l'API (on gère le cas où elle n'est pas dispo)
try:
    from totalsegmentator.python_api import totalsegmentator
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


def extract_patient_id(ct_path: Path) -> str:
    """
    Extrait un ID patient de manière robuste.

    Stratégie :
    1. On prend le nom du fichier sans extension
    2. On enlève le suffixe _TDM si présent
    3. Sinon fallback sur le dossier parent

    Exemple :
    - patient123_TDM.nii.gz → patient123
    - subject_001_TDM.nii.gz → subject_001
    """

    name = ct_path.name.replace(".nii.gz", "")

    if name.endswith("_TDM"):
        return name[:-4]  # enlève "_TDM"

    # fallback si format inattendu
    return ct_path.parent.name


def run_totalseg_api(ct_file: Path, output_mask: Path):
    """
    Lance TotalSegmentator via l'API Python.
    """
    totalsegmentator(
        input_path=str(ct_file),
        output_path=str(output_mask),
        task="total",
        roi_subset=["breast_female_left", "breast_female_right"],
        ml=True
    )


def run_totalseg_cli(ct_file: Path, output_mask: Path, device: str, fast: bool, tmp_dir: Path):
    """
    Lance TotalSegmentator via ligne de commande.
    On utilise un dossier temporaire car la CLI produit souvent plusieurs fichiers.
    """

    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", str(ct_file),
        "-o", str(tmp_dir),
        "-ta", "breasts",
        "--device", device,
        "--statistics",
    ]

    if fast:
        cmd.insert(4, "--fast")

    subprocess.run(cmd, check=True)

    # On récupère le masque sein généré
    breast_files = list(tmp_dir.glob("*breast*.nii.gz"))

    if not breast_files:
        raise RuntimeError("Aucun fichier breast trouvé en sortie CLI.")

    shutil.move(str(breast_files[0]), output_mask)


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation des seins avec TotalSegmentator (API ou CLI)."
    )

    # Arguments principaux
    parser.add_argument("input_root", type=Path, help="Dossier racine des scans")
    parser.add_argument("output_root", type=Path, help="Dossier de sortie")

    # Mode d'exécution
    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default="api",
        help="Choix du mode : API Python ou ligne de commande"
    )

    # Options
    parser.add_argument("--device", default="gpu:0", help="gpu:0 ou cpu")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (CLI seulement)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip si déjà traité")
    parser.add_argument("--suffix", default="_TDM.nii.gz", help="Suffixe des fichiers CT")

    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Vérification API
    if args.mode == "api" and not API_AVAILABLE:
        print("❌ API TotalSegmentator non disponible. Passe en mode CLI.")
        sys.exit(1)

    # Recherche récursive des fichiers
    ct_files = sorted(input_root.rglob(f"*{args.suffix}"))

    if not ct_files:
        print(f"❌ Aucun fichier trouvé dans {input_root}")
        return

    print(f"🔍 {len(ct_files)} fichier(s) trouvé(s)\n")

    for ct_file in ct_files:
        patient_id = extract_patient_id(ct_file)

        output_mask = output_root / f"{patient_id}_breast_mask.nii.gz"

        # Skip si déjà existant
        if args.skip_existing and output_mask.exists():
            print(f"[SKIP] {patient_id}")
            continue

        print(f"[RUN ] {patient_id} ({ct_file.name})")

        try:
            if args.mode == "api":
                run_totalseg_api(ct_file, output_mask)

            elif args.mode == "cli":
                tmp_dir = output_root / f"{patient_id}_tmp"
                run_totalseg_cli(ct_file, output_mask, args.device, args.fast, tmp_dir)
                shutil.rmtree(tmp_dir, ignore_errors=True)

            print(f"[DONE] {output_mask}")

        except Exception as e:
            print(f"[FAIL] {patient_id} : {e}")

    print("\n Terminé.")


if __name__ == "__main__":
    main()

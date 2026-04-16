import os
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator

def segment_breasts_batch(input_folder: str, output_folder: str):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # On récupère tous les fichiers scanner (CT) .nii.gz
    ct_files = list(input_path.glob("*_TDM.nii.gz")) # suffixe à adapter si besoin

    print(f"--- Début du traitement de {len(ct_files)} patients ---")

    for ct_file in ct_files:
        patient_id = ct_file.name.split('_')[0]
        print(f"Traitement du patient : {patient_id}")

        # Le fichier de sortie pour ce patient
        output_mask = output_path / f"{patient_id}_TDM_breast_mask.nii.gz"

        try:
            # On appelle l'API de TotalSegmentator
            # On demande spécifiquement les seins (breast_female_left et breast_female_right)
            # 'task="total"' couvre les 117 organes dont les seins
            totalsegmentator(
                input_path=str(ct_file),
                output_path=str(output_mask),
                task="total",
                roi_subset=["breast_female_left", "breast_female_right"],
                ml=True # Utilise le Deep Learning
            )
            print(f"✅ Masque généré : {output_mask}")
        except Exception as e:
            print(f"❌ Erreur pour le patient {patient_id} : {e}")

if __name__ == "__main__":
    # A remplace par les vrais chemins
    MY_INPUT_CT = "./data/images_ct"
    MY_OUTPUT_MASKS = "./data/breast_masks"
    
    segment_breasts_batch(MY_INPUT_CT, MY_OUTPUT_MASKS)

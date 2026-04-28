import os
import pydicom
import SimpleITK as sitk
import shutil

#TAF : On hypothèse que les séries sont rangées dans dossiers distincs donc que pas possible d'avoir 2 séries dans  dossiers différents. Mais si jamais la réalité terrain dit autrement, il faut modiff le code
#TAF : S'assurer que phases des DCE sont dans des séries différentes
# IMPORTANT : Ce code ne gère pas la conversion en SUV val, ilf audra donc le faire à part

def get_dicom_metadata(dicom_dir: str) -> dict:
    """
    Scanne un dossier pour trouver un fichier DICOM valide et extraire ses métadonnées.
    C'est la carte d'identité de notre série d'images.
    """
    fichiers = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    
    # On teste les fichiers un par un jusqu'à trouver un vrai DICOM
    for f in fichiers:
        try:
            # stop_before_pixels=True évite de charger l'image lourde en mémoire, on lit juste le texte
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            return {
                "PatientID": str(getattr(ds, "PatientID", "UNKNOWN")),
                "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
                "SeriesDescription": str(getattr(ds, "SeriesDescription", "UNKNOWN")),
                "SeriesTime": str(getattr(ds, "SeriesTime", "000000")), # Utile pour trier les phases IRM
            }
        except Exception:
            # Ce n'est pas un fichier DICOM (ex: un .txt ou fichier caché), on passe au suivant
            continue
            
    return {} # Si on arrive ici, le dossier ne contient aucun DICOM


def convert_dicom_series_to_nifti(dicom_dir: str, output_path: str):
    """
    Convertit une série DICOM contenue dans un dossier vers un fichier NIfTI unique.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        return False

    # IMPORTANT : On fai tl'hypothèse qu'un dossier contient une seule série. SI les réalité du terrain disent autrement, faudra adapter le code
    # On prend la première série trouvée dans le dossier
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(dicom_names)

    try:
        image = reader.Execute()
        # Création des dossiers parents (ex: imgs/) si ce n'est pas déjà fait
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(image, output_path)
        return True
    except Exception as e:
        print(f"   [ERREUR] Conversion échouée pour {dicom_dir} : {e}")
        return False


def ingest_raw_dicoms(raw_data_root: str, out_mri_root: str, out_petct_root: str, dict_anonymisation: dict = None):
    """
    Parcourt RÉCURSIVEMENT les dossiers de l'hôpital.
    Route les données vers DEUX univers séparés (IRM vs PET/CT).
    """
    mri_phases_by_patient = {}

    print("--- 1. SCAN RÉCURSIF DES DOSSIERS ---")
    dossiers_series = []
    
    # os.walk descend automatiquement dans tous les sous-dossiers, peu importe la profondeur !
    for root, dirs, files in os.walk(raw_data_root):
        if files:
            # S'il y a des fichiers dans ce sous-dossier, on l'ajoute à notre liste d'inspection
            dossiers_series.append(root) #Logiquement si on a au moins 1 fichier dans un rep, il est pris, alors il faut veiller à ce que cela soit pris en compte

    print(f"{len(dossiers_series)} dossiers potentiels trouvés. Analyse des métadonnées...")

    for dossier in dossiers_series:
        meta = get_dicom_metadata(dossier)
        if not meta:
            continue # Pas de DICOM ici, on ignore

        vrai_id = meta["PatientID"]
        modality = meta["Modality"]
        description = meta["SeriesDescription"].upper() # En majuscules pour faciliter la recherche
        
        # --- ANONYMISATION ---
        patient_id = vrai_id
        if dict_anonymisation and vrai_id in dict_anonymisation:
            patient_id = dict_anonymisation[vrai_id]

        # --- ROUTAGE PET / CT ---
        if modality in ["PT", "CT"]:
            print(f"\n[{modality}] {description} (Patient: {patient_id})")
            
            if modality == "PT":
                # NOUVEAU : On ne convertit pas le PET ! 
                # On copie les DICOM bruts dans un dossier "TEP/" pour le script SUV converter et nii maker.
                tep_dicom_dir = os.path.join(out_petct_root, patient_id, "TEP")
                print(f" -> Copie des DICOM bruts PET vers : {tep_dicom_dir}")
                
                # shutil.copytree copie tout le contenu du dossier source vers la destination
                shutil.copytree(dossier, tep_dicom_dir, dirs_exist_ok=True)

            else: # CT
                # Le CT, par contre, n'a pas besoin de calcul SUV, on le convertit directement en NIfTI.
                imgs_dir = os.path.join(out_petct_root, patient_id, "imgs")
                os.makedirs(imgs_dir, exist_ok=True)
                
                out_path = os.path.join(imgs_dir, f"{patient_id}_TDM.nii.gz")
                print(f" -> Conversion CT vers : {out_path}")
                convert_dicom_series_to_nifti(dossier, out_path)

        # --- ROUTAGE IRM (AVEC FILTRE T1/DCE) ---
        elif modality == "MR":
            # NOUVEAU : On ne garde que si "T1" ou "DCE" est dans le nom de la série
            if "T1" in description or "DCE" in description:
                print(f"\n[IRM Validée] {description} (Patient: {patient_id})")
                
                if patient_id not in mri_phases_by_patient:
                    mri_phases_by_patient[patient_id] = []
                
                mri_phases_by_patient[patient_id].append({
                    "dicom_dir": dossier,
                    "time": meta["SeriesTime"],
                    "desc": description
                })
            else:
                # C'est un T2, FLAIR, Diffusion, etc. On jette.
                pass 
                # print(f" [IRM Rejetée - Pas T1] {description}")

        else:
            pass # Modalité ignorée (ex: Radiographie classique CR, Dose Report SR...)

    # --- TRAITEMENT ET TRI DES PHASES IRM ---
    print("\n--- 2. SAUVEGARDE ET TRI DES PHASES IRM ---")
    for patient_id, phases in mri_phases_by_patient.items():
        # Tri chronologique selon l'heure d'acquisition DICOM
        phases_triees = sorted(phases, key=lambda x: x["time"])
        
        # Destination : Univers IRM
        imgs_dir = os.path.join(out_mri_root, patient_id, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)

        for index, phase in enumerate(phases_triees):
            # Les IRMs prennent un index (_0000, _0001...) car nnU-Net les gère comme des canaux temporels
            out_path = os.path.join(imgs_dir, f"{patient_id}_{index:04d}.nii.gz")
            print(f" -> IRM Phase {index} ({phase['desc']}) vers : {out_path}")
            convert_dicom_series_to_nifti(phase["dicom_dir"], out_path)

    print("\n=== INGÉSTION, ROUTAGE ET CONVERSION TERMINÉS ===")

if __name__ == "__main__":
    # --- CONFIGURATION DES CHEMINS ---
    DOSSIER_DICOM_VRAC = "./data_hopital_brut"
    
    # NOUVEAU : Deux racines bien distinctes pour ne pas mélanger les projets
    PROJET_IRM_RACINE = "./Base_IRM"
    PROJET_PETCT_RACINE = "./Base_PETCT"
    
    # Dictionnaire d'anonymisation (facultatif)
    CORRESPONDANCES = {
        "JEAN_DUPONT_849": "DUKE_001",
        "MARIE_CURIE_112": "DUKE_002"
    }

    ingest_raw_dicoms(
        raw_data_root=DOSSIER_DICOM_VRAC, 
        out_mri_root=PROJET_IRM_RACINE, 
        out_petct_root=PROJET_PETCT_RACINE, 
        dict_anonymisation=CORRESPONDANCES
    )

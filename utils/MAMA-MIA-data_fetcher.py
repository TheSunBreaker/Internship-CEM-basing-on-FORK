import os
import shutil
import synapseclient
import synapseutils
import urllib3

# ==========================================
# PARAMÈTRES DE CONFIGURATION
# ==========================================

TOKEN_SYNAPSE = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjE1NzYxNSwiaWF0IjoxNzc2MTU3NjE1LCJqdGkiOiIzNTU2MSIsInN1YiI6IjM1ODQyODQifQ.Ki62gZeEVvsTzIIxJZMk3m3qHROj-Vzc6hdAso-z6SHRfqx4pIEuiMxHQxZdV3twTRAuhi9A5gPjhuAnxn2P_lrONCoBL5ADYN5RMCvgaa9TSLts9d6WdWL2qKr1lE5LVCPPxLhd_I9xX8Ag8LQgil03nRjxCknivsoSRfcSin-Ep9u_ExTvK4R2G7oX7Bb1xb820wFxWjC5dkQr_dfgmavieDm1-a5Yqfgh9yEqs8f4oeX6HTUd6B6363TyoHa81zwtUNeHxV1EaluprkcyApfRSp7c_qZK_cfb8EozunDMY6flrMaIkwOPiQY0dVUBlA7MPopZ0ctnBXn5CJMMww"

# Identifiants Synapse
ID_DOSSIER_IMAGES = "syn64871114"   # Id synapse dossier parent contenant les dossiers patients (ex: DUKE_001/)
ID_DOSSIER_MASQUES = "syn64871175"  # Id synapse dossier unique contenant toutes les vérités terrain (ex: DUKE_001.nii.gz)

# Dossier racine où l'arborescence finale sera créée
DOSSIER_SORTIE = "./mes_dossiers_nnunet"

# FLAG: Activer ou désactiver la vérification SSL
DESACTIVER_VERIFICATION_SSL = True


def main():
    print("=== DÉBUT DU TÉLÉCHARGEMENT ET FORMATAGE DES DONNÉES ===")

    # 1. Gestion de la connexion et du SSL
    if DESACTIVER_VERIFICATION_SSL:
        print("[INFO] Contournement de la vérification SSL activé.")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        syn = synapseclient.Synapse(skip_checks=True)
        syn._requests_session.verify = False
    else:
        print("[INFO] Vérification SSL standard activée.")
        syn = synapseclient.Synapse()

    print("\nConnexion à Synapse en cours...")
    syn.login(authToken=TOKEN_SYNAPSE)
    print("Connexion réussie !")

    # 2. Préparation des dossiers locaux
    dossier_temp_masques = os.path.join(DOSSIER_SORTIE, "_temp_masks")
    os.makedirs(dossier_temp_masques, exist_ok=True)

    # 3. ÉTAPE A : Téléchargement groupé de TOUTES les vérités terrain
    # On télécharge tout le dossier des masques dans un dossier temporaire pour faire le tri ensuite.
    print("\n[ÉTAPE 1/3] Téléchargement de toutes les vérités terrain (Ground Truth)...")
    synapseutils.syncFromSynapse(syn, ID_DOSSIER_MASQUES, path=dossier_temp_masques)
    print("Vérités terrain téléchargées.")

    # 4. ÉTAPE B : Récupérer la liste de tous les patients
    print("\n[ÉTAPE 2/3] Récupération de la liste des patients...")
    dossiers_patients_generateur = syn.getChildren(parent=ID_DOSSIER_IMAGES, includeTypes=['folder'])
    dossiers_patients = list(dossiers_patients_generateur)  # On convertit le générateur en liste pour avoir la taille
    
    total_patients = len(dossiers_patients)
    print(f"{total_patients} dossiers patients trouvés sur Synapse.")

    # Statistiques pour le résumé final
    patients_valides = 0
    patients_ignores = 0

    # 5. ÉTAPE C : Boucle de traitement par patient
    print("\n[ÉTAPE 3/3] Alignement des images et des masques par patient...")
    
    for index, dossier in enumerate(dossiers_patients):
        patient_id = dossier['name']  # Exemple : "DUKE_001"
        
        # On définit le nom attendu pour le masque de ce patient
        nom_masque_attendu = f"{patient_id}.nii.gz"
        chemin_masque_temp = os.path.join(dossier_temp_masques, nom_masque_attendu)

        print(f"\n--- Traitement [{index + 1}/{total_patients}] : {patient_id} ---")

        # VÉRIFICATION CRUCIALE : Le masque existe-t-il pour ce patient ?
        if not os.path.exists(chemin_masque_temp):
            print(f"[ATTENTION] Vérité terrain introuvable pour {patient_id} (recherché: {nom_masque_attendu}).")
            print("-> Patient ignoré (nnU-Net a besoin des deux).")
            patients_ignores += 1
            continue

        print(f"[OK] Masque trouvé pour {patient_id}. Création de la structure...")

        # Création de la structure attendue par le script de préparation du dossier de train pour nnUnet
        # Exemple : ./mes_dossiers_nnunet/DUKE_001/imgs/ et ./mes_dossiers_nnunet/DUKE_001/mask/
        chemin_patient_racine = os.path.join(DOSSIER_SORTIE, patient_id)
        chemin_imgs = os.path.join(chemin_patient_racine, "imgs")
        chemin_mask = os.path.join(chemin_patient_racine, "mask")
        
        os.makedirs(chemin_imgs, exist_ok=True)
        os.makedirs(chemin_mask, exist_ok=True)

        # Déplacement du masque depuis le dossier temporaire vers son dossier définitif
        chemin_masque_definitif = os.path.join(chemin_mask, nom_masque_attendu)
        shutil.move(chemin_masque_temp, chemin_masque_definitif)

        # Téléchargement des images (les différentes phases IRM) directement dans le dossier "imgs"
        print(f"Téléchargement des images IRM de {patient_id} dans le dossier 'imgs'...")
        synapseutils.syncFromSynapse(syn, dossier['id'], path=chemin_imgs)
        
        patients_valides += 1

    # 6. NETTOYAGE
    print("\nNettoyage des fichiers temporaires...")
    if os.path.exists(dossier_temp_masques):
        # On supprime le dossier temporaire. S'il reste des masques dedans, 
        # c'est qu'ils n'avaient pas de dossier image correspondant sur Synapse.
        shutil.rmtree(dossier_temp_masques)

    # 7. RÉSUMÉ
    print("\n===========================================")
    print("TERMINÉ !")
    print(f"Total des patients analysés : {total_patients}")
    print(f"Patients formatés avec succès : {patients_valides}")
    print(f"Patients ignorés (masque manquant) : {patients_ignores}")
    print(f"Les données sont prêtes dans : {os.path.abspath(DOSSIER_SORTIE)}")
    print("===========================================")

if __name__ == "__main__":
    main()

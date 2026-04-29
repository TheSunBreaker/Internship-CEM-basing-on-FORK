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
ID_DOSSIER_IMAGES = "syn64871114"  # Dossier parent des dossiers patients
ID_DOSSIER_MASQUES = "syn64871175" # Dossier contenant tous les .nii.gz

# Dossier racine cible
DOSSIER_SORTIE = "./mes_dossiers_nnunet"

# FLAG: Activer ou désactiver la vérification SSL
DESACTIVER_VERIFICATION_SSL = True

def main():
    print("=== DÉBUT DU TÉLÉCHARGEMENT ET FORMATAGE OPTIMISÉ ===")

    # 1. Gestion de la connexion et du SSL
    if DESACTIVER_VERIFICATION_SSL:
        print("[INFO] Contournement de la vérification SSL activé.")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        syn = synapseclient.Synapse(skip_checks=True)
        syn._requests_session.verify = False
    else:
        syn = synapseclient.Synapse()

    print("\nConnexion à Synapse...")
    syn.login(authToken=TOKEN_SYNAPSE)
    print("Connexion réussie !")

    # 2. Préparation des dossiers
    # Dossier temporaire pour stocker TOUS les masques d'un coup
    dossier_temp_masques = os.path.join(DOSSIER_SORTIE, "_temp_masks")
    os.makedirs(dossier_temp_masques, exist_ok=True)

    # 3. ÉTAPE OPTIMISÉE : Téléchargement groupé des masques (Batch Download)
    # C'est ici que l'on gagne beaucoup de temps en évitant les requêtes unitaires
    print("\n[ÉTAPE 1/3] Téléchargement groupé de TOUTES les vérités terrain (Optimisé)...")
    synapseutils.syncFromSynapse(syn, ID_DOSSIER_MASQUES, path=dossier_temp_masques)
    
    # On crée un index local des masques réellement téléchargés
    masques_disponibles = os.listdir(dossier_temp_masques)
    print(f"Indexation terminée : {len(masques_disponibles)} masques prêts en local.")

    # 4. Récupération de la liste des patients
    print("\n[ÉTAPE 2/3] Récupération de la liste des patients...")
    dossiers_patients = list(syn.getChildren(parent=ID_DOSSIER_IMAGES, includeTypes=['folder']))
    total_patients = len(dossiers_patients)
    print(f"{total_patients} dossiers patients trouvés sur Synapse.")

    patients_valides = 0
    patients_ignores = 0

    # 5. Boucle de traitement (Déplacement local + Téléchargement images)
    print("\n[ÉTAPE 3/3] Alignement des images et des masques...")
    
    for index, dossier in enumerate(dossiers_patients):
        patient_id = dossier['name']
        nom_masque_attendu = f"{patient_id}.nii.gz"
        chemin_masque_source = os.path.join(dossier_temp_masques, nom_masque_attendu)

        print(f"\n--- [{index + 1}/{total_patients}] {patient_id} ---")

        # Vérification locale (pas d'appel API ici, très rapide)
        if nom_masque_attendu not in masques_disponibles:
            print(f"[SKIP] Masque introuvable localement pour {patient_id}.")
            patients_ignores += 1
            continue

        # Création de l'arborescence finale
        chemin_patient_racine = os.path.join(DOSSIER_SORTIE, patient_id)
        chemin_imgs = os.path.join(chemin_patient_racine, "imgs")
        chemin_mask = os.path.join(chemin_patient_racine, "mask")
        
        os.makedirs(chemin_imgs, exist_ok=True)
        os.makedirs(chemin_mask, exist_ok=True)

        # OPTIMISATION : Déplacement local au lieu de téléchargement Synapse
        chemin_masque_dest = os.path.join(chemin_mask, nom_masque_attendu)
        shutil.move(chemin_masque_source, chemin_masque_dest)
        print(f"-> Masque aligné (déplacement local)")

        # Téléchargement des images (obligatoire via API car spécifiques à chaque dossier patient)
        print(f"-> Téléchargement des images sources...")
        synapseutils.syncFromSynapse(syn, dossier['id'], path=chemin_imgs)
        
        patients_valides += 1

    # 6. NETTOYAGE
    print("\nNettoyage du dossier temporaire...")
    if os.path.exists(dossier_temp_masques):
        shutil.rmtree(dossier_temp_masques)

    # 7. RÉSUMÉ
    print("\n" + "="*40)
    print("TRAITEMENT TERMINÉ")
    print(f"Patients formatés : {patients_valides}")
    print(f"Patients ignorés  : {patients_ignores}")
    print(f"Localisation : {os.path.abspath(DOSSIER_SORTIE)}")
    print("="*40)

if __name__ == "__main__":
    main()

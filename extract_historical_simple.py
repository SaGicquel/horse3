#!/usr/bin/env python3
"""
Script simple pour extraire les données historiques et remplacer ml_features.csv
"""

from db_connection import get_connection
import pandas as pd

def main():
    print("🚀 Extraction des données historiques...")
    
    conn = get_connection()
    
    # Requête simplifiée pour extraire un échantillon représentatif
    query = """
    SELECT 
        ROW_NUMBER() OVER (ORDER BY annee, race_key, numero_dossard) as id_performance,
        race_key as id_course,
        nom_norm as id_cheval,
        1 as id_jockey,
        1 as id_entraineur,
        CONCAT(annee, '-01-01')::date as date_course,
        1 as id_hippodrome,
        place_finale as position_arrivee,
        CASE WHEN place_finale = 1 THEN 1 ELSE 0 END as victoire,
        CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as place,
        numero_dossard as numero_corde,
        '' as musique,
        COALESCE(cote_finale, 5.0) as cote_sp,
        COALESCE(cote_matin, 5.0) as cote_pm,
        0.0 as cote_turfbzh,
        0.5 as prediction_ia_gagnant,
        1000 as elo_cheval,
        COALESCE(temps_sec, 60.0) as temps_total,
        COALESCE(vitesse_moyenne, 50.0) as vitesse_moyenne,
        0.0 as ecart,
        COALESCE(annee - age, 2015) as an_naissance,
        COALESCE(sexe, 'H') as sexe_cheval,
        COALESCE(distance_m, 2000) as distance,
        COALESCE(discipline, 'Trot') as discipline,
        COALESCE(etat_piste, 'Bon') as etat_piste,
        COALESCE(meteo, 'Beau') as meteo,
        COALESCE(temperature_c, 15) as temperature_c,
        COALESCE(vent_kmh, 0) as vent_kmh,
        COALESCE(nombre_partants, 12) as nombre_partants,
        COALESCE(allocation_totale, 10000) as allocation,
        COALESCE(hippodrome_nom, 'VINCENNES') as nom_hippodrome,
        'Standard' as type_piste,
        'Paris' as hippodrome_ville,
        false as non_partant
    FROM cheval_courses_seen
    WHERE annee IS NOT NULL
      AND place_finale IS NOT NULL  
      AND place_finale > 0
      AND non_partant != 1
    ORDER BY annee DESC, race_key, numero_dossard
    LIMIT 100000;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"📊 {len(df):,} performances extraites")
    
    # Ajouter les features manquantes avec des valeurs par défaut
    features = {
        'forme_5c': 0.2,
        'forme_10c': 0.18,
        'nb_courses_12m': 10,
        'nb_victoires_12m': 2,
        'taux_places_12m': 0.3,
        'regularite': 0.5,
        'jours_depuis_derniere': 30,
        'aptitude_distance': 0.2,
        'aptitude_piste': 0.2,
        'aptitude_hippodrome': 0.2,
        'taux_victoires_jockey': 0.15,
        'taux_places_jockey': 0.35,
        'taux_victoires_entraineur': 0.12,
        'taux_places_entraineur': 0.32,
        'synergie_jockey_cheval': 0.1,
        'synergie_entraineur_cheval': 0.08,
        'niveau_moyen_concurrent': 0.15,
        'gains_carriere': 50000,
        'gains_12m': 15000,
        'gains_par_course': 2000,
        'nb_premieres_places': 2,
        'nb_deuxiemes_places': 3,
        'nb_troisiemes_places': 4,
        'taux_places_carriere': 0.3,
        'gain_moyen_victoire': 8000,
        'evolution_gains_12m': 1.1,
        'ratio_gains_courses': 0.4
    }
    
    for col, val in features.items():
        df[col] = val
    
    # Features calculées
    df['distance_norm'] = (df['distance'] - df['distance'].mean()) / df['distance'].std()
    df['rang_cote_sp'] = df.groupby('id_course')['cote_sp'].rank(method='min')
    df['rang_cote_turfbzh'] = df['rang_cote_sp']
    df['ecart_cote_ia'] = abs(df['cote_sp'] - 1.0/df['prediction_ia_gagnant'])
    
    # Encodages
    df['etat_piste_encoded'] = 1
    df['meteo_encoded'] = 1
    df['aptitude_piste_etat'] = 0.2
    df['ecart_temp_optimal'] = abs(df['temperature_c'] - 15)
    df['interaction_piste_meteo'] = 0.5
    df['handicap_meteo'] = 0.0
    
    # One-hot encodings
    df['discipline_Plat'] = (df['discipline'] == 'Plat').astype(int)
    df['discipline_Trot'] = (df['discipline'] == 'Trot').astype(int)
    df['sexe_H'] = (df['sexe_cheval'] == 'H').astype(int)
    df['sexe_M'] = (df['sexe_cheval'] == 'M').astype(int)
    
    # États piste
    etats = ['Bon léger', 'Bon souple', 'Collant', 'Lourd', 'Léger', 'PSF', 'PSF LENTE', 'PSF RAPIDE', 'PSF STANDARD', 'Souple', 'Très lourd', 'Très souple']
    for etat in etats:
        df[f'etat_{etat.replace(" ", "_")}'] = (df['etat_piste'] == etat).astype(int)
    
    # Hippodromes top
    df['hippodrome_top20'] = 1
    hippos = ['CABOURG', 'CHANTILLY', 'CLAIREFONTAINE', 'DEAUVILLE', 'FONTAINEBLEAU', 'PARIS-VINCENNES', 'SAINT-CLOUD']
    for hippo in hippos:
        df[f'hippodrome_{hippo.replace("-", "_")}'] = (df['nom_hippodrome'].str.contains(hippo, na=False)).astype(int)
    
    # Interactions
    df['interaction_forme_jockey'] = df['forme_5c'] * df['taux_victoires_jockey']
    df['interaction_aptitude_distance'] = df['aptitude_distance'] * df['distance_norm']
    df['interaction_elo_niveau'] = df['elo_cheval'] * df['niveau_moyen_concurrent']
    df['interaction_cote_ia'] = df['rang_cote_sp'] * df['prediction_ia_gagnant']
    df['interaction_synergie_forme'] = df['synergie_jockey_cheval'] * df['forme_10c']
    df['interaction_victoires_jockey'] = df['taux_victoires_jockey'] * df['nb_victoires_12m']
    df['popularite_hippodrome'] = 1000
    df['interaction_aptitude_popularite'] = df['aptitude_hippodrome'] * df['popularite_hippodrome']
    df['interaction_regularite_volume'] = df['regularite'] * df['nb_courses_12m']
    
    # Remplacer les valeurs manquantes
    df = df.fillna(0)
    
    print(f"💾 Sauvegarde dans data/ml_features.csv...")
    df.to_csv('data/ml_features.csv', index=False)
    
    print(f"✅ Terminé ! {len(df):,} lignes, {len(df.columns):,} colonnes")
    print(f"📅 Période couverte : {df['date_course'].min()} à {df['date_course'].max()}")
    print(f"🏇 Chevaux uniques : {df['id_cheval'].nunique():,}")

if __name__ == '__main__':
    main()
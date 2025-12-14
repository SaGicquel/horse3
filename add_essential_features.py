#!/usr/bin/env python3
"""
Script léger pour ajouter uniquement les features ML essentielles
Traitement par petits chunks pour éviter de surcharger le terminal
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

def add_essential_features_safe():
    """Ajoute les features essentielles par petits chunks"""
    
    print("🔄 Début du traitement par chunks...")
    
    # Lire le fichier par chunks de 5000 lignes
    chunk_size = 5000
    input_file = 'data/ml_features_complete.csv'
    output_file = 'data/ml_features_essential.csv'
    
    # Vérifier que le fichier existe
    if not Path(input_file).exists():
        print(f"❌ Fichier {input_file} introuvable")
        return
    
    # Traitement chunk par chunk
    first_chunk = True
    total_processed = 0
    
    for chunk_df in pd.read_csv(input_file, chunksize=chunk_size):
        print(f"📊 Traitement du chunk {total_processed//chunk_size + 1} ({len(chunk_df)} lignes)")
        
        # Ajouter features essentielles
        chunk_enhanced = add_features_to_chunk(chunk_df)
        
        # Sauvegarder le chunk
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        
        chunk_enhanced.to_csv(output_file, mode=mode, header=header, index=False)
        
        total_processed += len(chunk_df)
        first_chunk = False
        
        print(f"✅ Chunk sauvegardé. Total: {total_processed} lignes")
        time.sleep(0.1)  # Pause pour éviter la surcharge
    
    print(f"🎉 Terminé ! {total_processed} performances traitées")
    print(f"📄 Fichier créé: {output_file}")

def add_features_to_chunk(df):
    """Ajoute les features essentielles à un chunk"""
    
    # 1. Distance normalisée (feature critique)
    df['distance_norm'] = df['distance'] / 1000.0
    
    # 2. Nombre de partants (si pas déjà présent)
    if 'nombre_partants' not in df.columns:
        df['nombre_partants'] = 10  # Valeur par défaut
    
    # 3. Cote logarithmique (améliore les modèles)
    df['cote_log'] = np.log1p(df['cote_sp'].fillna(10.0))
    
    # 4. Position relative de la corde
    df['corde_relative'] = df['numero_corde'] / df['nombre_partants']
    
    # 5. Indicateur de favori (cote < 5)
    df['is_favori'] = (df['cote_sp'] < 5.0).astype(int)
    
    # 6. Indicateur de longue distance (> 2000m)
    df['is_longue_distance'] = (df['distance'] > 2000).astype(int)
    
    # 7. Année normalisée
    df['annee_norm'] = (df['annee'] - 2020) / 5.0
    
    return df

if __name__ == "__main__":
    add_essential_features_safe()
# -*- coding: utf-8 -*-
"""
Chargement CSV IFCE dans core.horse/core.horse_ifce avec normalisation et filtrage âge.
Colonnes CSV réelles: RACE,SEXE,ROBE,DATE_DE_NAISSANCE,PAYS_DE_NAISSANCE,NOM,CHE_COCONSO,DATE_DE_DECES
Filtres: vivant (DATE_DE_DECES vide), non-consommation (CHE_COCONSO='N'), âge course (2-15 ans).
"""
import csv
import re
from datetime import date, datetime
from typing import Dict, Any, Optional
from lib.normalize import normalize_name, normalize_country, birth_year_from_date
from db_connection import get_connection


def parse_date_to_iso(date_str: Optional[str]) -> Optional[str]:
    """Convertit DD/MM/YYYY en YYYY-MM-DD pour PostgreSQL DATE."""
    if not date_str or not date_str.strip():
        return None
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', date_str.strip())
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"
    return None


def infer_columns(header: list[str]) -> Dict[str, str]:
    hmap = {h.lower().replace('_','').replace(' ',''): h for h in header}
    def pick(*cands):
        for c in cands:
            key = c.lower().replace('_','').replace(' ','')
            if key in hmap:
                return hmap[key]
        return None
    return {
        'name': pick('NOM','Nom','name','Cheval','libelle'),
        'birth': pick('DATE_DE_NAISSANCE','DateNaissance','date_naissance','birth','naissance'),
        'sex': pick('SEXE','Sexe','sex'),
        'country': pick('PAYS_DE_NAISSANCE','Pays','country','pays_naissance','pays'),
        'race': pick('RACE','Race','breed'),
        'id': pick('ID','Id','ifce_id','identifiant'),
        'coconso': pick('CHE_COCONSO','coconso','consommation'),
        'deces': pick('DATE_DE_DECES','date_deces','deces','death')
    }


def age_ok(birth_year: int, today_year: int, min_age=2, max_age=15) -> bool:
    """Chevaux en âge de courir (2 à 15 ans)."""
    if not birth_year:
        return False  # si date naissance inconnue, on rejette
    age = today_year - birth_year
    return (age >= min_age) and (age <= max_age)


def is_racing_horse(coconso: str, deces: str) -> bool:
    """Filtre: vivant (pas de date décès) et non destiné à consommation (CHE_COCONSO='N')."""
    # Vivant = DATE_DE_DECES vide
    if deces and deces.strip():
        return False
    # Non consommation = CHE_COCONSO == 'N'
    if coconso and coconso.strip().upper() == 'O':  # 'O' = destiné consommation
        return False
    return True


def load_csv(path: str):
    con = get_connection()
    cur = con.cursor()

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = infer_columns(header)
        idx = {k: (header.index(v) if v in header else None) for k,v in cols.items()}

        today_year = date.today().year
        inserted = 0
        skipped_age = 0
        skipped_dead = 0
        skipped_coconso = 0

        for row in reader:
            def val(key):
                i = idx.get(key)
                return row[i].strip() if (i is not None and i < len(row)) else None

            name_raw = val('name')
            if not name_raw:
                continue
            name_norm = normalize_name(name_raw)
            birth_raw = val('birth')
            birth_date = parse_date_to_iso(birth_raw)  # conversion DD/MM/YYYY -> YYYY-MM-DD
            birth_year = birth_year_from_date(birth_raw)
            sex = (val('sex') or '').upper() or None
            country = normalize_country(val('country'))
            breed = val('race')
            ifce_id = val('id')
            coconso = val('coconso')
            deces = val('deces')

            # Filtrage: vivant, non-consommation
            if not is_racing_horse(coconso, deces):
                if deces and deces.strip():
                    skipped_dead += 1
                elif coconso and coconso.strip().upper() == 'O':
                    skipped_coconso += 1
                continue

            # Filtrage: âge course
            if not age_ok(birth_year, today_year, min_age=2, max_age=15):
                skipped_age += 1
                continue

            # upsert core.horse
            cur.execute(
                """
                INSERT INTO core.horse(name_norm, birth_date, sex, breed, country, source, external_id)
                VALUES (core.normalize_name(%s), %s, %s, %s, %s, 'IFCE', %s)
                ON CONFLICT (name_norm, birth_date, sex) DO UPDATE
                SET breed = COALESCE(EXCLUDED.breed, core.horse.breed),
                    country = COALESCE(EXCLUDED.country, core.horse.country)
                RETURNING id
                """,
                (name_raw, birth_date, sex, breed, country, ifce_id)
            )
            horse_id = cur.fetchone()[0]

            # upsert IFCE link (utiliser nom+date comme ID synthétique si pas d'ID réel)
            synthetic_id = ifce_id or f"{name_norm}_{birth_date or 'UNKNOWN'}_{sex or 'U'}"
            cur.execute(
                """
                INSERT INTO core.horse_ifce(horse_id, ifce_id, raw)
                VALUES (%s, %s, to_jsonb(%s::text))
                ON CONFLICT (horse_id) DO UPDATE SET ifce_id = EXCLUDED.ifce_id
                """,
                (horse_id, synthetic_id, name_raw)
            )

            inserted += 1

    con.commit()
    con.close()
    print(f"✅ IFCE import terminé: {inserted} chevaux ajoutés/mis à jour")
    print(f"   Ignorés: {skipped_dead} décédés, {skipped_coconso} consommation, {skipped_age} âge hors limites")


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'fichier-des-equides.csv'
    load_csv(path)

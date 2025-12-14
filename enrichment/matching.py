# -*- coding: utf-8 -*-
"""
Module de matching PMU ↔ IFCE
Associe les chevaux PMU aux enregistrements IFCE avec plusieurs stratégies
"""

# import sqlite3 # REMOVED: Migration to PostgreSQL
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enrichment.normalization import normalize_name, extract_birth_year, normalize_country


@dataclass
class MatchResult:
    """Résultat d'un matching"""
    ifce_horse_key: Optional[int]
    match_stage: str  # 'A' (strict), 'B' (souple), 'C' (fuzzy), 'none', 'ambiguous'
    confidence: float  # 0.0 - 1.0
    details: str = ""


class HorseMatcher:
    """
    Gestionnaire de matching PMU ↔ IFCE avec stratégies multiples.
    
    Stratégies :
        A (strict) : name_norm + birth_year + sex/country
        B (souple) : name_norm + (birth_year OU country)
        C (fuzzy) : FTS + Levenshtein ≤ 2 + birth_year ±1
    """
    
    def __init__(self, db_conn: Any, enable_fuzzy: bool = False):
        """
        Args:
            db_conn: Connexion PostgreSQL (psycopg2)
            enable_fuzzy: Activer le matching fuzzy (stage C) - REQUIRES pg_trgm
        """
        self.conn = db_conn
        self.enable_fuzzy = enable_fuzzy
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Crée les index nécessaires si absents"""
        cur = self.conn.cursor()
        
        # Index sur ifce_horses
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ifce_name_norm 
            ON ifce_horses(name_norm)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ifce_birth_year 
            ON ifce_horses(birth_year)
        """)
        
        # Index sur pmu_horses
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_pmu_name_norm 
            ON pmu_horses(name_norm)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_pmu_ifce 
            ON pmu_horses(ifce_horse_key)
        """)
        
        # Index sur horse_aliases
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_aliases_pmu 
            ON horse_aliases(pmu_name_norm)
        """)
        
        self.conn.commit()
    
    def match_horse(self, name: str, birth_year: Optional[int] = None,
                   sex: Optional[str] = None, country: Optional[str] = None) -> MatchResult:
        """
        Tente de matcher un cheval PMU avec IFCE selon les stratégies A → B → C.
        
        Args:
            name: Nom du cheval
            birth_year: Année de naissance (optionnel)
            sex: Sexe H/M/F (optionnel)
            country: Code pays (optionnel)
            
        Returns:
            MatchResult avec ifce_horse_key et match_stage
        """
        name_norm = normalize_name(name)
        if not name_norm:
            return MatchResult(None, 'none', 0.0, "Nom invalide")
        
        cur = self.conn.cursor()
        
        # 1. Vérifier cache alias
        cur.execute("""
            SELECT ifce_horse_key FROM horse_aliases 
            WHERE pmu_name_norm = %s
        """, (name_norm,))
        row = cur.fetchone()
        if row:
            return MatchResult(row[0], 'A', 1.0, "Cache alias")
        
        # 2. Stratégie A : strict (name + birth_year + sex/country)
        result = self._match_strategy_a(cur, name_norm, birth_year, sex, country)
        if result.ifce_horse_key:
            return result
        
        # 3. Stratégie B : souple (name + birth_year OU country)
        result = self._match_strategy_b(cur, name_norm, birth_year, country)
        if result.ifce_horse_key:
            return result
        
        # 4. Stratégie C : fuzzy (optionnel)
        if self.enable_fuzzy:
            result = self._match_strategy_c(cur, name_norm, birth_year)
            if result.ifce_horse_key:
                return result
        
        return MatchResult(None, 'none', 0.0, "Aucun match trouvé")
    
    def _match_strategy_a(self, cur: sqlite3.Cursor, name_norm: str,
                         birth_year: Optional[int], sex: Optional[str],
                         country: Optional[str]) -> MatchResult:
        """
        Stratégie A (strict) : name_norm + birth_year + sex/country
        """
        if not birth_year:
            return MatchResult(None, 'none', 0.0, "Stratégie A : birth_year manquant")
        
        # Requête avec birth_year obligatoire + sex OU country si dispos
        conditions = ["name_norm = %s", "birth_year = %s"]
        params = [name_norm, birth_year]
        
        # Stratégie A nécessite au moins sex OU country en plus de name+year
        if not sex and not country:
            return MatchResult(None, 'none', 0.0, "Stratégie A : sex ou country manquant")
        
        if sex:
            conditions.append("sex = %s")
            params.append(sex)
        
        if country:
            country_norm = normalize_country(country)
            conditions.append("country = %s")
            params.append(country_norm)
        
        query = f"""
            SELECT horse_key, name, birth_year, sex, country 
            FROM ifce_horses 
            WHERE {' AND '.join(conditions)}
        """
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        if len(rows) == 0:
            return MatchResult(None, 'none', 0.0, "Stratégie A : aucun candidat")
        elif len(rows) == 1:
            return MatchResult(rows[0][0], 'A', 1.0, f"Strict match : {rows[0][1]}")
        else:
            # Ambiguïté : plusieurs matches stricts
            candidates = ', '.join([f"{r[1]} ({r[2]})" for r in rows])
            return MatchResult(None, 'ambiguous', 0.5, f"Ambiguïté A : {candidates}")
    
    def _match_strategy_b(self, cur: sqlite3.Cursor, name_norm: str,
                         birth_year: Optional[int], country: Optional[str]) -> MatchResult:
        """
        Stratégie B (souple) : name_norm + (birth_year OU country)
        """
        # Cas 1 : name + birth_year (sans country)
        if birth_year:
            cur.execute("""
                SELECT horse_key, name, birth_year, country 
                FROM ifce_horses 
                WHERE name_norm = %s AND birth_year = %s
            """, (name_norm, birth_year))
            rows = cur.fetchall()
            
            if len(rows) == 1:
                return MatchResult(rows[0][0], 'B', 0.85, f"Match name+year : {rows[0][1]}")
            elif len(rows) > 1:
                candidates = ', '.join([f"{r[1]} ({r[2]}, {r[3]})" for r in rows])
                return MatchResult(None, 'ambiguous', 0.4, f"Ambiguïté B : {candidates}")
        
        # Cas 2 : name + country (sans birth_year)
        if country:
            country_norm = normalize_country(country)
            cur.execute("""
                SELECT horse_key, name, birth_year, country 
                FROM ifce_horses 
                WHERE name_norm = %s AND country = %s
            """, (name_norm, country_norm))
            rows = cur.fetchall()
            
            if len(rows) == 1:
                return MatchResult(rows[0][0], 'B', 0.75, f"Match name+country : {rows[0][1]}")
            elif len(rows) > 1:
                candidates = ', '.join([f"{r[1]} ({r[2]}, {r[3]})" for r in rows])
                return MatchResult(None, 'ambiguous', 0.3, f"Ambiguïté B : {candidates}")
        
        return MatchResult(None, 'none', 0.0, "Stratégie B : critères insuffisants")
    
    def _match_strategy_c(self, cur: Any, name_norm: str,
                         birth_year: Optional[int]) -> MatchResult:
        """
        Stratégie C (fuzzy) : FTS + Levenshtein ≤ 2 + birth_year ±1
        Nécessite FTS activé sur ifce_horses.
        """
        if not birth_year:
            return MatchResult(None, 'none', 0.0, "Stratégie C : birth_year requis")
        
        # TODO: Adapter pour PostgreSQL (pg_trgm)
        return MatchResult(None, 'none', 0.0, "Stratégie C : Non implémenté pour PostgreSQL")

        # # Vérifier si FTS existe
        # cur.execute("""
        #     SELECT name FROM sqlite_master 
        #     WHERE type='table' AND name='ifce_horses_fts'
        # """)
        # if not cur.fetchone():
        #     return MatchResult(None, 'none', 0.0, "Stratégie C : FTS non activé")
        
        # # Recherche FTS
        # cur.execute("""
        #     SELECT h.horse_key, h.name, h.birth_year, h.name_norm
        #     FROM ifce_horses h
        #     JOIN ifce_horses_fts fts ON h.horse_key = fts.rowid
        #     WHERE ifce_horses_fts MATCH ?
        #     AND h.birth_year BETWEEN ? AND ?
        #     LIMIT 10
        # """, (name_norm, birth_year - 1, birth_year + 1))
        
        # candidates = cur.fetchall()
        # if not candidates:
        #     return MatchResult(None, 'none', 0.0, "Stratégie C : aucun candidat fuzzy")
        
        # # Calcul Levenshtein sur candidats
        # best_match = None
        # best_distance = 999
        
        # for cand in candidates:
        #     dist = levenshtein_distance(name_norm, cand[3])
        #     if dist <= 2 and dist < best_distance:
        #         best_distance = dist
        #         best_match = cand
        
        # if best_match:
        #     confidence = 0.6 if best_distance == 0 else (0.5 if best_distance == 1 else 0.4)
        #     return MatchResult(
        #         best_match[0], 
        #         'C', 
        #         confidence,
        #         f"Fuzzy match (dist={best_distance}) : {best_match[1]}"
        #     )
        
        # return MatchResult(None, 'none', 0.0, "Stratégie C : distance Levenshtein > 2")
    
    def cache_alias(self, pmu_name: str, ifce_horse_key: int):
        """
        Enregistre un alias validé dans le cache.
        
        Args:
            pmu_name: Nom du cheval côté PMU
            ifce_horse_key: Clé du cheval IFCE correspondant
        """
        name_norm = normalize_name(pmu_name)
        if not name_norm:
            return
        
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO horse_aliases (pmu_name_norm, ifce_horse_key)
            VALUES (%s, %s)
            ON CONFLICT (pmu_name_norm) DO UPDATE SET
                ifce_horse_key = EXCLUDED.ifce_horse_key
        """, (name_norm, ifce_horse_key))
        self.conn.commit()
    
    def generate_match_report(self) -> Dict[str, int]:
        """
        Génère un rapport KPI du matching.
        
        Returns:
            Dict avec compteurs par match_stage : {'A': 150, 'B': 30, 'C': 10, 'none': 5, 'ambiguous': 2}
        """
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT match_stage, COUNT(*) 
            FROM pmu_horses 
            WHERE match_stage IS NOT NULL
            GROUP BY match_stage
        """)
        
        report = {stage: count for stage, count in cur.fetchall()}
        
        # Compléter avec zéros
        for stage in ['A', 'B', 'C', 'none', 'ambiguous']:
            report.setdefault(stage, 0)
        
        return report
    
    def update_pmu_horse_match(self, pmu_horse_id: int, match_result: MatchResult):
        """
        Met à jour un cheval PMU avec le résultat du matching.
        
        Args:
            pmu_horse_id: ID du cheval dans pmu_horses
            match_result: Résultat du matching
        """
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE pmu_horses
            SET ifce_horse_key = %s,
                match_stage = %s,
                match_confidence = %s
            WHERE pmu_horse_id = %s
        """, (match_result.ifce_horse_key, match_result.match_stage, 
              match_result.confidence, pmu_horse_id))
        self.conn.commit()


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calcule la distance de Levenshtein entre deux chaînes.
    
    Args:
        s1, s2: Chaînes à comparer
        
    Returns:
        Distance (nombre de modifications minimales)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Coût insertion, suppression, substitution
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


if __name__ == '__main__':
    # Tests rapides
    print("Tests Levenshtein :")
    print("-" * 60)
    test_pairs = [
        ("ELEGANT DAVRIL", "ELEGANT D'AVRIL"),
        ("BLACK SAXON", "BLACK SAXONE"),
        ("SAINT MARTIN", "SAINTMARTIN"),
        ("BONJOUR", "BONJOURNO"),
    ]
    for s1, s2 in test_pairs:
        dist = levenshtein_distance(s1, s2)
        print(f"{s1:20} <-> {s2:20} = {dist}")

# -*- coding: utf-8 -*-
"""Matching IFCE ↔ PMU chevaux via normalisation + trigram + Levenshtein.

Règles:
- Pré-match exact (name_norm égal et année de naissance ±1) => confirmed (method=exact)
- Trigram pg_trgm: si score ≥ threshold_confirm => confirmed (method=trigram)
- Fallback Levenshtein (rapidfuzz ratio) si trigram < confirm et ratio ≥ 95 => confirmed (method=levenshtein)
- Sinon: enregistrer les top candidats ≥ threshold_ambiguous comme ambiguous, avec evidences.
"""
from db_connection import get_connection
from lib.normalize import normalize_name
from typing import List, Tuple, Optional
from psycopg2.extras import Json

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def fetch_ifce(cur):
    cur.execute("""
        SELECT h.id, h.name_norm, h.birth_date
        FROM core.horse h
        JOIN core.horse_ifce i ON i.horse_id = h.id
    """)
    return cur.fetchall()


def fetch_pmu(cur):
    cur.execute("""
        SELECT h.id, h.name_norm, h.birth_date
        FROM core.horse h
        JOIN core.horse_pmu p ON p.horse_id = h.id
    """)
    return cur.fetchall()


def similarity(cur, a: str, b: str) -> float:
    cur.execute("SELECT similarity(%s,%s)", (a, b))
    return cur.fetchone()[0] or 0.0


def year(val: Optional[object]) -> Optional[int]:
    try:
        return getattr(val, 'year', None) or int(str(val)[:4])
    except Exception:
        return None


def levenshtein(a: str, b: str) -> Optional[float]:
    if not fuzz:
        return None
    # ratio 0..100
    return float(fuzz.ratio(a, b))


def run_matching(threshold_confirm=0.90, threshold_ambiguous=0.85, max_ambiguous_candidates: int = 3):
    con = get_connection()
    cur = con.cursor()

    ifce_list = fetch_ifce(cur)
    pmu_list = fetch_pmu(cur)

    # Index PMU par première lettre du nom normalisé pour accélérer
    pmu_by_first_letter = {}
    for pid, pname, pbirth in pmu_list:
        first = pname[0] if pname else '?'
        pmu_by_first_letter.setdefault(first, []).append((pid, pname, pbirth))

    matched = 0
    ambiguous = 0

    for iid, iname, ibirth in ifce_list:
        ib = year(ibirth)
        first = iname[0] if iname else '?'
        
        # Limiter recherche aux PMU avec même première lettre
        candidates_pmu = pmu_by_first_letter.get(first, [])

        # 1) Pré-match exact par nom normalisé et année ±1
        exact_candidates: List[Tuple[str, str, Optional[int]]] = [
            (pid, pname, year(pbirth)) for (pid, pname, pbirth) in candidates_pmu
            if pname == iname and (
                ib is None or year(pbirth) is None or abs(ib - year(pbirth)) <= 1
            )
        ]
        if len(exact_candidates) == 1:
            pid, pname, pb = exact_candidates[0]
            cur.execute(
                """
                INSERT INTO aux.match_horse(ifce_horse_id, pmu_horse_id, score, method, status, evidence)
                VALUES (%s,%s,%s,'exact','confirmed', jsonb_build_object('ifce_name',%s,'pmu_name',%s,'birth_delta',%s))
                ON CONFLICT (ifce_horse_id, pmu_horse_id) DO NOTHING
                """,
                (iid, pid, 1.0, iname, pname, None if ib is None or pb is None else abs(ib - pb))
            )
            matched += 1
            continue

        # 2) Trigram scoring global avec filtre année ±1
        scored: List[Tuple[str, str, Optional[int], float]] = []
        for pid, pname, pbirth in candidates_pmu:
            pb = year(pbirth)
            if ib is not None and pb is not None and abs(ib - pb) > 1:
                continue
            sc = similarity(cur, iname, pname)
            scored.append((pid, pname, pb, sc))

        scored.sort(key=lambda t: t[3], reverse=True)
        best = scored[0] if scored else None

        # 3) Confirmation par trigram
        if best and best[3] >= threshold_confirm:
            pid, pname, pb, sc = best
            cur.execute(
                """
                INSERT INTO aux.match_horse(ifce_horse_id, pmu_horse_id, score, method, status, evidence)
                VALUES (%s,%s,%s,'trigram','confirmed', jsonb_build_object('ifce_name',%s,'pmu_name',%s,'birth_delta',%s))
                ON CONFLICT (ifce_horse_id, pmu_horse_id) DO NOTHING
                """,
                (iid, pid, sc, iname, pname, None if ib is None or pb is None else abs(ib - pb))
            )
            matched += 1
            continue

        # 4) Fallback Levenshtein si dispo
        if best and fuzz:
            pid, pname, pb, sc = best
            lv = levenshtein(iname, pname)
            if lv is not None and lv >= 95:
                cur.execute(
                    """
                    INSERT INTO aux.match_horse(ifce_horse_id, pmu_horse_id, score, method, status, evidence)
                    VALUES (%s,%s,%s,'levenshtein','confirmed', jsonb_build_object('ifce_name',%s,'pmu_name',%s,'levenshtein',%s,'trigram',%s))
                    ON CONFLICT (ifce_horse_id, pmu_horse_id) DO NOTHING
                    """,
                    (iid, pid, float(lv)/100.0, iname, pname, lv, sc)
                )
                matched += 1
                continue

        # 5) Ambiguïtés: enregistrer top N >= threshold_ambiguous
        ambs = [t for t in scored if t[3] >= threshold_ambiguous][:max_ambiguous_candidates]
        if ambs:
            for pid, pname, pb, sc in ambs:
                cur.execute(
                    """
                    INSERT INTO aux.match_horse(ifce_horse_id, pmu_horse_id, score, method, status, evidence)
                    VALUES (%s,%s,%s,'trigram','ambiguous', jsonb_build_object('ifce_name',%s,'pmu_name',%s))
                    ON CONFLICT (ifce_horse_id, pmu_horse_id) DO NOTHING
                    """,
                    (iid, pid, sc, iname, pname)
                )
            ambiguous += 1

    con.commit()
    con.close()
    print(f"✅ Matching terminé: {matched} confirmés, {ambiguous} ambigus")


if __name__ == '__main__':
    run_matching()

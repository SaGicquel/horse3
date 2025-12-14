# -*- coding: utf-8 -*-
"""Matching IFCE ↔ PMU chevaux rapide avec batch et progression.
Version optimisée pour grands volumes (197k IFCE × 694 PMU).
"""
from db_connection import get_connection
from lib.normalize import normalize_name
from typing import List, Tuple, Optional
from psycopg2.extras import Json

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def year(val: Optional[object]) -> Optional[int]:
    try:
        return getattr(val, 'year', None) or int(str(val)[:4])
    except Exception:
        return None


def levenshtein(a: str, b: str) -> Optional[float]:
    if not fuzz:
        return None
    return float(fuzz.ratio(a, b))


def run_matching_fast(threshold_confirm=0.90, threshold_ambiguous=0.85, batch_size=1000):
    con = get_connection()
    cur = con.cursor()

    # Charger PMU en mémoire
    cur.execute("""
        SELECT h.id, h.name_norm, h.birth_date
        FROM core.horse h
        JOIN core.horse_pmu p ON p.horse_id = h.id
    """)
    pmu_list = cur.fetchall()
    print(f"PMU: {len(pmu_list)} chevaux")

    # Index PMU par première lettre
    pmu_by_first = {}
    for pid, pname, pbirth in pmu_list:
        first = pname[0] if pname else '?'
        pmu_by_first.setdefault(first, []).append((pid, pname, pbirth))

    # Charger IFCE par batch
    cur.execute("SELECT count(*) FROM core.horse_ifce")
    total_ifce = cur.fetchone()[0]
    print(f"IFCE: {total_ifce} chevaux à matcher")

    matched = 0
    ambiguous = 0
    processed = 0

    offset = 0
    while True:
        cur.execute("""
            SELECT h.id, h.name_norm, h.birth_date
            FROM core.horse h
            JOIN core.horse_ifce i ON i.horse_id = h.id
            ORDER BY h.id
            LIMIT %s OFFSET %s
        """, (batch_size, offset))
        batch = cur.fetchall()
        if not batch:
            break

        for iid, iname, ibirth in batch:
            ib = year(ibirth)
            first = iname[0] if iname else '?'
            candidates = pmu_by_first.get(first, [])

            # Exact match
            exact = [(pid, pname, year(pbirth)) for (pid, pname, pbirth) in candidates
                     if pname == iname and (ib is None or year(pbirth) is None or abs(ib - year(pbirth)) <= 1)]
            if len(exact) == 1:
                pid, pname, pb = exact[0]
                cur.execute("""
                    INSERT INTO aux.match_horse(ifce_horse_id, pmu_horse_id, score, method, status, evidence)
                    VALUES (%s,%s,%s,'exact','confirmed', jsonb_build_object('ifce_name',%s,'pmu_name',%s))
                    ON CONFLICT (ifce_horse_id, pmu_horse_id) DO NOTHING
                """, (iid, pid, 1.0, iname, pname))
                matched += 1
                continue

            # Trigram top match (limité aux candidats même première lettre)
            if candidates:
                best_pid, best_pname, best_pb, best_sc = None, None, None, 0.0
                for pid, pname, pbirth in candidates[:50]:  # limite max 50 comparaisons par cheval
                    pb = year(pbirth)
                    if ib is not None and pb is not None and abs(ib - pb) > 1:
                        continue
                    cur.execute("SELECT similarity(%s,%s)", (iname, pname))
                    sc = cur.fetchone()[0] or 0.0
                    if sc > best_sc:
                        best_sc = sc
                        best_pid, best_pname, best_pb = pid, pname, pb

                if best_sc >= threshold_confirm:
                    cur.execute("""
                        INSERT INTO aux.match_horse(ifce_horse_id, pmu_horse_id, score, method, status, evidence)
                        VALUES (%s,%s,%s,'trigram','confirmed', jsonb_build_object('ifce_name',%s,'pmu_name',%s))
                        ON CONFLICT (ifce_horse_id, pmu_horse_id) DO NOTHING
                    """, (iid, best_pid, best_sc, iname, best_pname))
                    matched += 1

        processed += len(batch)
        con.commit()
        print(f"  Traité {processed}/{total_ifce} ({100*processed//total_ifce}%) - Matches: {matched}")
        offset += batch_size

    con.close()
    print(f"✅ Matching terminé: {matched} confirmés, {ambiguous} ambigus")


if __name__ == '__main__':
    run_matching_fast()

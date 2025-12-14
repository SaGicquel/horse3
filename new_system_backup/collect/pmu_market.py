# -*- coding: utf-8 -*-
"""Collecte snapshots de marché (cotes) simplifiée.

Hypothèse: les cotes pré-course sont disponibles dans le payload 'course' ou un endpoint futur.
Ici on relit le payload 'course' (déjà stocké) et enregistre un snapshot artificiel si des champs de cotes existent.
"""
from datetime import datetime
from db_connection import get_connection


def collect_market_snapshot(date_iso: str):
    con = get_connection()
    cur = con.cursor()
    # Chercher payloads course
    cur.execute("SELECT key, payload FROM stg.raw_payloads WHERE source='PMU_API' AND endpoint='course' AND key LIKE %s", (f"{date_iso}|R%|C%",))
    rows = cur.fetchall()
    captured_at = datetime.utcnow()
    for key, payload in rows:
        parts = payload.get('participants') or []
        if not parts:
            continue
        # retrouver race_id
        _, r, c = key.split('|')[0], key.split('|')[1], key.split('|')[2]
        num_r = int(r[1:])
        num_c = int(c[1:])
        cur.execute("""
            SELECT r.id FROM fact.race r JOIN fact.meeting m ON r.meeting_id = m.id
            WHERE m.date=%s AND m.num_reunion=%s AND r.num_course=%s
        """, (date_iso, num_r, num_c))
        rr = cur.fetchone()
        if not rr:
            continue
        race_id = rr[0]
        for p in parts:
            num_pmu = p.get('numPmu') or p.get('numero')
            cote = p.get('cote') or p.get('odd') or p.get('rapportProbable')
            if cote is None:
                continue
            try:
                cote_val = float(cote)
            except Exception:
                continue
            cur.execute("""
                INSERT INTO fact.market_snapshot(race_id, captured_at, type_pari, num_pmu, cote, overround, source)
                VALUES (%s,%s,'SIMPLE_GAGNANT',%s,%s,NULL,'PMU_API')
                ON CONFLICT DO NOTHING
            """, (race_id, captured_at, num_pmu, cote_val))
    con.commit()
    con.close()


if __name__ == '__main__':
    import sys
    from datetime import date
    date_iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    collect_market_snapshot(date_iso)

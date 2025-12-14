# -*- coding: utf-8 -*-
"""Rapport qualité basique sous forme Markdown en BDD."""
from datetime import datetime
from db_connection import get_connection
from psycopg2.extras import Json


def generate_quality_report(date_iso: str) -> str:
    con = get_connection()
    cur = con.cursor()

    # Taux de match confirmés
    cur.execute("SELECT count(*) FROM aux.match_horse WHERE status='confirmed'")
    confirmed = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM aux.match_horse WHERE status='ambiguous'")
    ambiguous = cur.fetchone()[0]

    # IFCE non appariés
    cur.execute("""
        SELECT count(*)
        FROM core.horse_ifce i
        LEFT JOIN aux.match_horse m ON m.ifce_horse_id = i.horse_id AND m.status='confirmed'
        WHERE m.id IS NULL
    """)
    ifce_orphans = cur.fetchone()[0]

    # Courses sans participants
    cur.execute("""
        SELECT count(*) FROM (
            SELECT r.id, count(p.id) as np
            FROM fact.race r
            LEFT JOIN fact.participant p ON p.race_id = r.id
            GROUP BY r.id
        ) t WHERE t.np = 0
    """)
    races_without_participants = cur.fetchone()[0]

    # Distribution similitudes
    cur.execute("SELECT percentile_disc(ARRAY[0.5,0.8,0.9,0.95,0.99]) WITHIN GROUP (ORDER BY score) FROM aux.match_horse")
    row = cur.fetchone()
    percentiles = row[0] if row and row[0] is not None else []

    md = []
    md.append(f"# Rapport qualité {date_iso}\n")
    md.append(f"Généré: {datetime.utcnow().isoformat()}Z\n")
    md.append("\n## Appariement cheval IFCE ↔ PMU\n")
    md.append(f"- Confirmés: {confirmed}\n")
    md.append(f"- Ambigus: {ambiguous}\n")
    md.append(f"- IFCE non appariés: {ifce_orphans}\n")

    md.append("\n## Races / Participants\n")
    md.append(f"- Courses sans participants: {races_without_participants}\n")

    md.append("\n## Similarités (percentiles)\n")
    md.append(f"- 50/80/90/95/99%: {percentiles}\n")

    content = "".join(md)

    # Enregistrer dans BDD (optionnel)
    cur.execute(
        "INSERT INTO stg.raw_payloads(source, endpoint, key, payload) VALUES ('REPORT','quality',%s,%s::jsonb) ON CONFLICT (source, endpoint, key) DO UPDATE SET payload=EXCLUDED.payload, fetched_at=NOW()",
        (date_iso, Json({"content": content}))
    )
    con.commit()
    con.close()
    print(content)
    return content


if __name__ == '__main__':
    from datetime import date
    generate_quality_report(date.today().isoformat())

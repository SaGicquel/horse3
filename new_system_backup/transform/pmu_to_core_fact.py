# -*- coding: utf-8 -*-
"""
Transformation des payloads PMU (staging) vers core/fact.
Étapes:
 - programme_jour / programme_reunion -> fact.meeting + core.hippodrome
 - course -> fact.race
 - participants -> fact.participant (sans horse_id au début)
 - performances -> enrichissements temps / red_km quand dispo
 - rapports -> fact.payout
 - puis création/liaison core.horse_pmu pour chevaux PMU et tentative de match
"""
from db_connection import get_connection
from lib.normalize import normalize_name
from psycopg2.extras import Json


def upsert_hippodrome(cur, hippo: dict):
    code = hippo.get('code') or hippo.get('codeHippodrome') or hippo.get('libelleCourt')
    name = hippo.get('libelleLong') or hippo.get('libelleCourt') or code
    if not code:
        return None
    cur.execute(
        """
        INSERT INTO core.hippodrome(code, name, country, region, zone)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (code) DO UPDATE SET name = COALESCE(EXCLUDED.name, core.hippodrome.name)
        RETURNING code
        """,
        (code, name, hippo.get('pays'), hippo.get('region'), hippo.get('zone'))
    )
    return cur.fetchone()[0]


# Mappings vers ENUMs
DISCIPLINE_MAP = {
    'ATTELE': 'TROT_A',
    'TROT_ATTELE': 'TROT_A',
    'MONTE': 'TROT_M',
    'TROT_MONTE': 'TROT_M',
    'PLAT': 'PLAT',
    'HAIES': 'HAIES',
    'STEEPLE': 'STEEPLE',
    'CROSS': 'CROSS',
}


def map_discipline(x: str):
    if not x:
        return None
    x = str(x).upper()
    return DISCIPLINE_MAP.get(x)


def map_corde(x: str):
    if not x:
        return None
    x = str(x).upper()
    if x in ('G', 'GAUCHE', 'CORDE_GAUCHE'):
        return 'G'
    if x in ('D', 'DROITE', 'CORDE_DROITE'):
        return 'D'
    return None


def map_type_depart(x: str):
    if not x:
        return None
    x = str(x).upper()
    if x in ('AUTOSTART', 'VOLTE', 'LIGNE'):
        return x
    return None


def transform_program(date_iso: str):
    con = get_connection()
    cur = con.cursor()

    # Réunions
    cur.execute("SELECT payload FROM stg.raw_payloads WHERE source='PMU_API' AND endpoint='programme_reunion' AND key LIKE %s", (f"{date_iso}|R%",))
    for (payload,) in cur.fetchall():
        hippo = payload.get('hippodrome') or {}
        code = upsert_hippodrome(cur, hippo)
        num_r = payload.get('numOfficiel') or payload.get('numExterneReunion')
        tz = payload.get('timezoneOffset')
        cur.execute(
            """
            INSERT INTO fact.meeting(date, hippo_code, num_reunion, timezone_offset, flags)
            VALUES (%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (date, hippo_code, num_reunion) DO UPDATE SET timezone_offset = EXCLUDED.timezone_offset, flags = EXCLUDED.flags
            RETURNING id
            """,
            (date_iso, code, num_r, tz, Json(payload.get('flags') or {}))
        )
        meeting_id = cur.fetchone()[0]

        # Courses listées dans le payload
        for c in (payload.get('courses') or []):
            num_c = c.get('numOrdre') or c.get('numExterne') or c.get('num')
            cur.execute(
                """
                INSERT INTO fact.race(meeting_id, num_course, libelle, discipline, specialite, distance_m, corde, type_depart, type_piste, classe, conditions, heure_depart, allocation_total, status, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, to_timestamp((%s)/1000), %s,%s,%s::jsonb)
                ON CONFLICT (meeting_id, num_course) DO UPDATE SET libelle=EXCLUDED.libelle, distance_m=EXCLUDED.distance_m, raw=EXCLUDED.raw
                """,
                (
                    meeting_id, num_c, c.get('libelle'), map_discipline(c.get('discipline')), c.get('specialite'), c.get('distance'),
                    map_corde(c.get('corde')), map_type_depart(c.get('typeDepart')), c.get('typePiste'), c.get('categorieParticularite'),
                    c.get('conditions'), c.get('heureDepart'), c.get('montantPrix') or c.get('allocation'), c.get('statut'), Json(c)
                )
            )

    con.commit()
    con.close()


def transform_course_level(date_iso: str):
    con = get_connection()
    cur = con.cursor()

    # Courses détaillées (endpoint 'course')
    cur.execute("SELECT key, payload FROM stg.raw_payloads WHERE source='PMU_API' AND endpoint='course' AND key LIKE %s", (f"{date_iso}|R%|C%",))
    for key, payload in cur.fetchall():
        _, r, c = key.split('|')[0], key.split('|')[1], key.split('|')[2]
        num_r = int(r[1:])
        num_c = int(c[1:])

        # retrouver meeting_id
        cur.execute("""
            SELECT m.id FROM fact.meeting m
            JOIN core.hippodrome h ON h.code = m.hippo_code
            WHERE m.date=%s AND m.num_reunion=%s
        """, (date_iso, num_r))
        row = cur.fetchone()
        if not row:
            continue
        meeting_id = row[0]
        # upsert race raw
        cur.execute("""
            INSERT INTO fact.race(meeting_id, num_course, raw)
            VALUES (%s,%s,%s::jsonb)
            ON CONFLICT (meeting_id, num_course) DO UPDATE SET raw=EXCLUDED.raw
        """, (meeting_id, num_c, Json(payload)))

        # Incidents extraction
        # race_id for this course
        cur.execute("SELECT id FROM fact.race WHERE meeting_id=%s AND num_course=%s", (meeting_id, num_c))
        race_row = cur.fetchone()
        race_id = race_row[0] if race_row else None
        if not race_id:
            continue
        incidents = payload.get('incidents') or []
        # map participant num -> participant id for linking
        cur.execute("""
            SELECT p.num_pmu, p.id FROM fact.participant p
            JOIN fact.race r ON r.id = p.race_id
            WHERE r.meeting_id=%s AND r.num_course=%s
        """, (meeting_id, num_c))
        num_to_pid = {n: pid for (n, pid) in cur.fetchall() if n is not None}
        for inc in incidents:
            itype = inc.get('type') or 'UNKNOWN'
            nums = inc.get('numeroParticipants') or []
            description = inc.get('description')
            severity = inc.get('severity')
            for n in nums:
                cur.execute("""
                    INSERT INTO fact.incident(race_id, participant_id, participant_num, type, code, description, severity, occurred_at, raw)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, NULL, %s::jsonb)
                """, (race_id, num_to_pid.get(n), n, itype, itype, description, severity, Json(inc)))

    con.commit()
    con.close()


def transform_participants(date_iso: str):
    con = get_connection()
    cur = con.cursor()

    cur.execute("SELECT key, payload FROM stg.raw_payloads WHERE source='PMU_API' AND endpoint='participants' AND key LIKE %s", (f"{date_iso}|R%|C%",))
    for key, payload in cur.fetchall():
        _, r, c = key.split('|')[0], key.split('|')[1], key.split('|')[2]
        num_r = int(r[1:])
        num_c = int(c[1:])
        # race_id
        cur.execute("""
            SELECT r.id FROM fact.race r JOIN fact.meeting m ON r.meeting_id = m.id
            WHERE m.date=%s AND m.num_reunion=%s AND r.num_course=%s
        """, (date_iso, num_r, num_c))
        row = cur.fetchone()
        if not row:
            continue
        race_id = row[0]

        parts = payload.get('participants') or []
        for p in parts:
            num_pmu = p.get('numPmu') or p.get('numero')
            name = p.get('nom')
            name_norm = normalize_name(name)
            pmu_horse_id = p.get('idCheval')
            # créer cheval PMU en core (sans doublon)
            cur.execute(
                """
                INSERT INTO core.horse(name_norm, source, external_id)
                VALUES (core.normalize_name(%s), 'PMU', %s)
                ON CONFLICT (name_norm, birth_date, sex) DO NOTHING
                RETURNING id
                """,
                (name, pmu_horse_id)
            )
            row_h = cur.fetchone()
            horse_id = row_h[0] if row_h else None
            if horse_id and pmu_horse_id:
                # Upsert sur pmu_id pour éviter violation unique quand même cheval déjà vu
                cur.execute(
                    """
                    INSERT INTO core.horse_pmu(horse_id, pmu_id, raw)
                    VALUES (%s,%s,%s::jsonb)
                    ON CONFLICT (pmu_id) DO UPDATE SET horse_id=EXCLUDED.horse_id, raw=EXCLUDED.raw
                    """,
                    (horse_id, str(pmu_horse_id), Json(p))
                )

            # upsert participant
            # Equipement parsing basique
            equip_tokens = []
            if p.get('oeilleres'): equip_tokens.append('OEILLERES')
            if p.get('masque'): equip_tokens.append('MASQUE')
            if p.get('protections'): equip_tokens.append('PROTECTIONS')
            equipement = ','.join(equip_tokens) if equip_tokens else None
            # Déferrage mapping simple (code déjà normalisé possible)
            deferrage_code = p.get('deferrage') or p.get('codeDeferrage')
            if deferrage_code:
                deferrage_code = deferrage_code.upper()
                if deferrage_code not in ('D4','D2','DA','DP','FERRE'):
                    deferrage_code = None
            cur.execute(
                """
                INSERT INTO fact.participant(race_id, horse_id, num_pmu, stalle, ligne_autostart, num_autostart, equipement, deferrage, poids_kg, handicap_distance, proprietaire, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (race_id, num_pmu) DO UPDATE SET equipement=EXCLUDED.equipement, deferrage=EXCLUDED.deferrage, raw=EXCLUDED.raw
                """,
                (
                    race_id, horse_id, num_pmu,
                    p.get('numeroStalle') or p.get('stalle'),
                    p.get('ligne') or p.get('ligneAutostart'),
                    p.get('numeroAutostart') or p.get('numAutostart'),
                    equipement,
                    deferrage_code,
                    p.get('poids'),
                    p.get('handicapDistance') or p.get('distanceSupplementaire'),
                    (p.get('proprietaire') or {}).get('nom') if isinstance(p.get('proprietaire'), dict) else p.get('proprietaire'),
                    Json(p)
                )
            )

    con.commit()
    con.close()


def transform_rapports(date_iso: str):
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT key, payload FROM stg.raw_payloads WHERE source='PMU_API' AND endpoint='rapports' AND key LIKE %s", (f"{date_iso}|R%|C%",))
    for key, payload in cur.fetchall():
        _, r, c = key.split('|')[0], key.split('|')[1], key.split('|')[2]
        num_r = int(r[1:])
        num_c = int(c[1:])
        cur.execute("""
            SELECT r.id FROM fact.race r JOIN fact.meeting m ON r.meeting_id = m.id
            WHERE m.date=%s AND m.num_reunion=%s AND r.num_course=%s
        """, (date_iso, num_r, num_c))
        row = cur.fetchone()
        if not row:
            continue
        race_id = row[0]

        items = payload if isinstance(payload, list) else []
        for it in items:
            raw_type = it.get('typePari')
            # mapping vers ENUM pool_type_enum
            type_map = {
                'SIMPLE_GAGNANT':'SIMPLE_GAGNANT','SIMPLE_PLACE':'SIMPLE_PLACE',
                'COUPLE_GAGNANT':'COUPLE_GAGNANT','COUPLE_PLACE':'COUPLE_PLACE',
                'DEUX_SUR_QUATRE':'DEUX_SUR_QUATRE','MULTI':'MULTI','TIERCE':'TIERCE',
                'QUARTE_PLUS':'QUARTE_PLUS','QUINTE_PLUS':'QUINTE_PLUS','TIC_TROIS':'TIC_TROIS',
                'REPORT_PLUS':'REPORT_PLUS','E_SIMPLE_GAGNANT':'E_SIMPLE_GAGNANT','E_SIMPLE_PLACE':'E_SIMPLE_PLACE',
                'E_COUPLE_GAGNANT':'E_COUPLE_GAGNANT','E_COUPLE_PLACE':'E_COUPLE_PLACE','E_DEUX_SUR_QUATRE':'E_DEUX_SUR_QUATRE',
                'E_MULTI':'E_MULTI','E_TIERCE':'E_TIERCE','E_QUARTE_PLUS':'E_QUARTE_PLUS','E_QUINTE_PLUS':'E_QUINTE_PLUS','E_REPORT_PLUS':'E_REPORT_PLUS'
            }
            type_pari = type_map.get(raw_type)  # ignorera TRIO ou autres non listés
            if not type_pari:
                continue
            rapports = it.get('rapports') or []
            for r in rapports:
                cur.execute(
                    """
                    INSERT INTO fact.payout(race_id, type_pari, combinaison, dividende, unite, gagnants, raw)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        race_id, type_pari, r.get('combinaison'),
                        r.get('dividende') or r.get('dividendePourUnEuro'),
                        r.get('dividendeUnite'), r.get('nombreGagnants'), Json(r)
                    )
                )

    con.commit()
    con.close()


if __name__ == '__main__':
    import sys
    from datetime import date
    date_iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    transform_program(date_iso)
    transform_course_level(date_iso)
    transform_participants(date_iso)
    transform_rapports(date_iso)

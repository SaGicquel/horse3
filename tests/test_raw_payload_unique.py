from db_connection import get_connection


def test_raw_payload_unique():
    con = get_connection()
    cur = con.cursor()
    # insérer un payload bidon
    cur.execute("""
        INSERT INTO stg.raw_payloads(source, endpoint, key, payload)
        VALUES ('TEST','endpoint','KEY1','{}'::jsonb)
        ON CONFLICT (source, endpoint, key) DO UPDATE SET payload='{"replaced":true}'::jsonb
        RETURNING payload
    """)
    row = cur.fetchone()
    assert row is not None
    con.commit()
    con.close()

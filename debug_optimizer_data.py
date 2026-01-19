import pandas as pd
from db_connection import get_connection
from web.backend.main import run_benter_head_for_date


def check_data():
    conn = get_connection()
    cur = conn.cursor()
    date_str = "2025-11-20"  # Random date in range

    print(f"Checking data for {date_str}")

    # Check Odds
    cur.execute(
        """
        SELECT cote_reference, cote_finale, nom_norm
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        AND cote_finale < 10
        LIMIT 5
    """,
        (date_str + "%",),
    )

    print("\n--- ODDS SAMPLE ---")
    for row in cur.fetchall():
        print(f"Ref: {row[0]}, Fin: {row[1]}, Name: {row[2]}")

    # Check Benter
    print("\n--- BENTER SAMPLE ---")
    res = run_benter_head_for_date(date_str, cur=cur)
    b_map = res.get("by_runner", {})
    if not b_map:
        print("Benter returned empty map.")
    else:
        k = list(b_map.keys())[0]
        print(f"Benter Key: {k}, Val: {b_map[k]}")

    conn.close()


if __name__ == "__main__":
    check_data()

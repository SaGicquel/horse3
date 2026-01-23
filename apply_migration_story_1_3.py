import sys
from db_connection import get_connection


def apply_migration():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        with open("sql/create_bet_tracking_table.sql", "r") as f:
            sql = f.read()

        cursor.execute(sql)
        conn.commit()
        print("✅ Migration applied successfully: bet_tracking table created.")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    apply_migration()

#!/usr/bin/env python3
import psycopg2
from datetime import date

conn = psycopg2.connect(
    host="localhost",
    port=54624,
    database="pmu_database",
    user="pmu_user",
    password="pmu_secure_password_2025",
)

cursor = conn.cursor()

# Vérifier aujourd'hui
today = date.today()
cursor.execute("SELECT COUNT(*) FROM courses WHERE date_course = %s", (today,))
count_today = cursor.fetchone()[0]
print(f"Courses pour aujourd'hui ({today}): {count_today}")

# Vérifier les chevaux pour aujourd'hui
cursor.execute(
    """
    SELECT COUNT(*)
    FROM chevaux
    WHERE course_id IN (SELECT id FROM courses WHERE date_course = %s)
""",
    (today,),
)
count_chevaux = cursor.fetchone()[0]
print(f"Chevaux pour aujourd'hui: {count_chevaux}")

# Prochaines dates
cursor.execute("""
    SELECT date_course, COUNT(*) as nb_courses
    FROM courses
    WHERE date_course >= CURRENT_DATE
    GROUP BY date_course
    ORDER BY date_course
    LIMIT 5
""")

print("\nProchaines dates de courses:")
for date_course, nb_courses in cursor.fetchall():
    print(f"  {date_course}: {nb_courses} courses")

cursor.close()
conn.close()

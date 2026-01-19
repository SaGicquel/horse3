from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

# Adaptation pour web/backend
try:
    from backend.database import get_cursor
    from backend.schemas import (
        CourseSummary,
        CourseDetail,
        PartantRead,
        ChevalSummary,
        HippodromeRead,
        RacesTodayResponse,
        RaceDetailResponse,
        PaginatedResponse,
    )
except ImportError:
    # Fallback pour execution depuis web/backend/
    from database import get_cursor
    from schemas import (
        CourseSummary,
        CourseDetail,
        PartantRead,
        ChevalSummary,
        HippodromeRead,
        RacesTodayResponse,
        RaceDetailResponse,
        PaginatedResponse,
    )

router = APIRouter(prefix="/api/races", tags=["races"])


@router.get("/today", response_model=RacesTodayResponse)
def get_races_today():
    today = date.today()

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id_course,
                c.date_course,
                c.heure_course,
                c.num_reunion,
                c.num_course,
                c.discipline,
                c.distance,
                c.nombre_partants,
                c.statut,
                h.nom_hippodrome as hippodrome_nom
            FROM courses c
            LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
            WHERE c.date_course = %s
            ORDER BY c.num_reunion, c.num_course
        """,
            (today,),
        )
        rows = cur.fetchall()

    courses = [CourseSummary(**dict(row)) for row in rows]

    return RacesTodayResponse(date=today, nombre_courses=len(courses), courses=courses)


@router.get("/date/{race_date}", response_model=RacesTodayResponse)
def get_races_by_date(race_date: date):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id_course,
                c.date_course,
                c.heure_course,
                c.num_reunion,
                c.num_course,
                c.discipline,
                c.distance,
                c.nombre_partants,
                c.statut,
                h.nom_hippodrome as hippodrome_nom
            FROM courses c
            LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
            WHERE c.date_course = %s
            ORDER BY c.num_reunion, c.num_course
        """,
            (race_date,),
        )
        rows = cur.fetchall()

    courses = [CourseSummary(**dict(row)) for row in rows]

    return RacesTodayResponse(date=race_date, nombre_courses=len(courses), courses=courses)


@router.get("/{race_id}", response_model=RaceDetailResponse)
def get_race_detail(race_id: str):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                c.*,
                h.id_hippodrome as h_id,
                h.nom_hippodrome,
                h.code_pmu as h_code_pmu,
                h.pays,
                h.type_piste,
                h.ville,
                h.region,
                h.latitude,
                h.longitude,
                h.altitude_m
            FROM courses c
            LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
            WHERE c.id_course = %s
        """,
            (race_id,),
        )
        course_row = cur.fetchone()

    if not course_row:
        raise HTTPException(status_code=404, detail=f"Course {race_id} not found")

    course_data = dict(course_row)

    hippodrome = None
    if course_data.get("h_id"):
        hippodrome = HippodromeRead(
            id_hippodrome=course_data["h_id"],
            nom_hippodrome=course_data["nom_hippodrome"],
            code_pmu=course_data["h_code_pmu"],
            pays=course_data.get("pays", "FR"),
            type_piste=course_data.get("type_piste"),
            ville=course_data.get("ville"),
            region=course_data.get("region"),
            latitude=course_data.get("latitude"),
            longitude=course_data.get("longitude"),
            altitude_m=course_data.get("altitude_m"),
        )

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                p.id_performance,
                p.numero_corde,
                p.numero_dossard,
                p.poids_porte,
                p.musique,
                p.deferre,
                p.oeilleres,
                p.cote_pm,
                p.cote_sp,
                p.cote_turfbzh,
                p.position_arrivee,
                p.place,
                p.non_partant,
                ch.id_cheval,
                ch.nom_cheval,
                ch.sexe_cheval,
                ch.an_naissance,
                ch.nombre_courses_total,
                ch.nombre_victoires_total,
                j.nom_complet as jockey_nom,
                e.nom_complet as entraineur_nom
            FROM performances p
            JOIN chevaux ch ON p.id_cheval = ch.id_cheval
            LEFT JOIN personnes j ON p.id_jockey = j.id_personne
            LEFT JOIN personnes e ON p.id_entraineur = e.id_personne
            WHERE p.id_course = %s
            ORDER BY p.numero_corde
        """,
            (race_id,),
        )
        partants_rows = cur.fetchall()

    partants = []
    for row in partants_rows:
        row_dict = dict(row)
        cheval = ChevalSummary(
            id_cheval=row_dict["id_cheval"],
            nom_cheval=row_dict["nom_cheval"],
            sexe_cheval=row_dict.get("sexe_cheval"),
            an_naissance=row_dict["an_naissance"],
            nombre_courses_total=row_dict.get("nombre_courses_total", 0),
            nombre_victoires_total=row_dict.get("nombre_victoires_total", 0),
        )

        partant = PartantRead(
            id_performance=row_dict["id_performance"],
            numero_corde=row_dict["numero_corde"],
            numero_dossard=row_dict.get("numero_dossard"),
            cheval=cheval,
            jockey_nom=row_dict.get("jockey_nom"),
            entraineur_nom=row_dict.get("entraineur_nom"),
            poids_porte=row_dict.get("poids_porte"),
            musique=row_dict.get("musique"),
            deferre=row_dict.get("deferre"),
            oeilleres=row_dict.get("oeilleres"),
            cote_pm=row_dict.get("cote_pm"),
            cote_sp=row_dict.get("cote_sp"),
            cote_turfbzh=row_dict.get("cote_turfbzh"),
            position_arrivee=row_dict.get("position_arrivee"),
            place=row_dict.get("place", False),
            non_partant=row_dict.get("non_partant", False),
        )
        partants.append(partant)

    course = CourseDetail(
        id_course=course_data["id_course"],
        date_course=course_data["date_course"],
        heure_course=course_data.get("heure_course"),
        num_reunion=course_data["num_reunion"],
        num_course=course_data["num_course"],
        discipline=course_data["discipline"],
        distance=course_data["distance"],
        allocation=course_data.get("allocation"),
        nombre_partants=course_data.get("nombre_partants"),
        id_hippodrome=course_data.get("id_hippodrome"),
        corde=course_data.get("corde"),
        etat_piste=course_data.get("etat_piste"),
        categorie_age=course_data.get("categorie_age"),
        sexe_condition=course_data.get("sexe_condition"),
        poids_condition=course_data.get("poids_condition"),
        meteo=course_data.get("meteo"),
        temperature_c=course_data.get("temperature_c"),
        vent_kmh=course_data.get("vent_kmh"),
        statut=course_data.get("statut", "PREVUE"),
        created_at=course_data.get("created_at"),
        updated_at=course_data.get("updated_at"),
        hippodrome=hippodrome,
        partants=partants,
    )

    return RaceDetailResponse(course=course)


@router.get("/", response_model=PaginatedResponse)
def list_races(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    discipline: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
):
    offset = (page - 1) * per_page

    conditions = []
    params = []

    if discipline:
        conditions.append("c.discipline = %s")
        params.append(discipline)
    if date_from:
        conditions.append("c.date_course >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("c.date_course <= %s")
        params.append(date_to)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) as total FROM courses c {where_clause}", params)
        total = cur.fetchone()["total"]

    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT
                c.id_course,
                c.date_course,
                c.heure_course,
                c.num_reunion,
                c.num_course,
                c.discipline,
                c.distance,
                c.nombre_partants,
                c.statut,
                h.nom_hippodrome as hippodrome_nom
            FROM courses c
            LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
            {where_clause}
            ORDER BY c.date_course DESC, c.num_reunion, c.num_course
            LIMIT %s OFFSET %s
        """,
            params + [per_page, offset],
        )
        rows = cur.fetchall()

    courses = [CourseSummary(**dict(row)) for row in rows]
    pages = (total + per_page - 1) // per_page

    return PaginatedResponse(total=total, page=page, per_page=per_page, pages=pages, items=courses)

"""
🏇 Schémas Pydantic - Modèles de données API
=============================================

Modèles Pydantic reflétant la structure de la base PostgreSQL.
Tables principales: courses, chevaux, performances, hippodromes, personnes.

Auteur: Task 2 - API Backend Core
"""

from datetime import date, time, datetime
from typing import Optional, List
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# MODÈLES DE BASE (Entités DB)
# ============================================================================


class HippodromeBase(BaseModel):
    """Hippodrome - lieu de course."""

    nom_hippodrome: str
    code_pmu: str
    pays: str = "FR"
    type_piste: Optional[str] = None
    ville: Optional[str] = None
    region: Optional[str] = None


class HippodromeRead(HippodromeBase):
    """Hippodrome avec ID (lecture)."""

    id_hippodrome: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class PersonneBase(BaseModel):
    """Jockey ou Entraîneur."""

    nom_complet: str
    type: str = Field(..., pattern="^(JOCKEY|ENTRAINEUR)$")
    code_pmu: Optional[str] = None


class PersonneRead(PersonneBase):
    """Personne avec ID (lecture)."""

    id_personne: int
    poids_jockey: Optional[int] = None
    indice_jockey: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChevalBase(BaseModel):
    """Cheval - données de base."""

    nom_cheval: str
    sexe_cheval: Optional[str] = Field(None, pattern="^[MHF]$")
    an_naissance: int
    robe: Optional[str] = None
    origine: Optional[str] = None
    nom_pere: Optional[str] = None
    nom_mere: Optional[str] = None
    proprietaire: Optional[str] = None


class ChevalRead(ChevalBase):
    """Cheval avec ID et stats (lecture)."""

    id_cheval: int
    code_pmu: Optional[str] = None
    nombre_courses_total: Optional[int] = 0
    nombre_victoires_total: Optional[int] = 0

    # Stats par discipline
    nombre_courses_trot: Optional[int] = None
    nombre_victoires_trot: Optional[int] = None
    nombre_courses_plat: Optional[int] = None
    nombre_victoires_plat: Optional[int] = None
    nombre_courses_obstacle: Optional[int] = None
    nombre_victoires_obstacle: Optional[int] = None

    # Gains
    gains_trot: Optional[int] = None
    gains_plat: Optional[int] = None
    gains_obstacle: Optional[int] = None

    # Forme récente
    forme_recente_30j: Optional[str] = None
    serie_victoires: Optional[int] = None
    serie_places: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ChevalSummary(BaseModel):
    """Résumé cheval pour affichage dans course."""

    id_cheval: int
    nom_cheval: str
    sexe_cheval: Optional[str] = None
    an_naissance: int
    nombre_courses_total: Optional[int] = 0
    nombre_victoires_total: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# PERFORMANCES (Participation d'un cheval à une course)
# ============================================================================


class PerformanceBase(BaseModel):
    """Performance - participation d'un cheval à une course."""

    numero_corde: int
    numero_dossard: Optional[int] = None
    poids_porte: Optional[int] = None
    musique: Optional[str] = None
    deferre: Optional[str] = None
    oeilleres: Optional[str] = None


class PerformanceRead(PerformanceBase):
    """Performance complète avec résultats."""

    id_performance: int
    id_course: str
    id_cheval: int
    id_jockey: Optional[int] = None
    id_entraineur: Optional[int] = None

    # Cotes
    cote_pm: Optional[float] = None
    cote_sp: Optional[float] = None
    cote_turfbzh: Optional[float] = None

    # Résultats (remplis après course)
    position_arrivee: Optional[int] = None
    place: bool = False
    ecart: Optional[str] = None
    disqualifie: bool = False
    non_partant: bool = False
    gain_course: Optional[int] = None
    rapport_gagnant: Optional[float] = None
    rapport_place: Optional[float] = None
    temps_total: Optional[float] = None
    vitesse_moyenne: Optional[float] = None

    # Prédictions IA
    prediction_ia_gagnant: Optional[float] = None
    elo_cheval: Optional[int] = None
    popularite: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class PartantRead(BaseModel):
    """Partant enrichi avec infos cheval/jockey pour affichage course."""

    # Identifiants
    id_performance: int
    numero_corde: int
    numero_dossard: Optional[int] = None

    # Cheval
    cheval: ChevalSummary

    # Jockey/Entraineur (noms seulement pour affichage)
    jockey_nom: Optional[str] = None
    entraineur_nom: Optional[str] = None

    # Équipement
    poids_porte: Optional[int] = None
    musique: Optional[str] = None
    deferre: Optional[str] = None
    oeilleres: Optional[str] = None

    # Cotes
    cote_pm: Optional[float] = None
    cote_sp: Optional[float] = None
    cote_turfbzh: Optional[float] = None

    # Résultats (si course terminée)
    position_arrivee: Optional[int] = None
    place: bool = False
    non_partant: bool = False

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# COURSES
# ============================================================================


class CourseBase(BaseModel):
    """Course - données de base."""

    date_course: date
    heure_course: Optional[time] = None
    num_reunion: int
    num_course: int
    discipline: str
    distance: int
    allocation: Optional[int] = None
    nombre_partants: Optional[int] = None


class CourseRead(CourseBase):
    """Course complète (lecture)."""

    id_course: str
    id_hippodrome: Optional[int] = None

    # Conditions
    corde: Optional[str] = None
    etat_piste: Optional[str] = None
    categorie_age: Optional[str] = None
    sexe_condition: Optional[str] = None
    poids_condition: Optional[str] = None

    # Météo
    meteo: Optional[str] = None
    temperature_c: Optional[float] = None
    vent_kmh: Optional[float] = None

    # Statut
    statut: str = "PREVUE"

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CourseWithHippodrome(CourseRead):
    """Course avec informations hippodrome."""

    hippodrome: Optional[HippodromeRead] = None

    model_config = ConfigDict(from_attributes=True)


class CourseDetail(CourseWithHippodrome):
    """Course complète avec tous les partants."""

    partants: List[PartantRead] = []

    model_config = ConfigDict(from_attributes=True)


class CourseSummary(BaseModel):
    """Résumé course pour liste."""

    id_course: str
    date_course: date
    heure_course: Optional[time] = None
    num_reunion: int
    num_course: int
    discipline: str
    distance: int
    nombre_partants: Optional[int] = None
    statut: str = "PREVUE"

    # Hippodrome (nom seulement)
    hippodrome_nom: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# RÉUNIONS (Groupement de courses par jour/hippodrome)
# ============================================================================


class ReunionSummary(BaseModel):
    """Résumé d'une réunion (ensemble de courses)."""

    date_reunion: date
    num_reunion: int
    hippodrome: Optional[HippodromeRead] = None
    nombre_courses: int
    courses: List[CourseSummary] = []

    model_config = ConfigDict(from_attributes=True)


class JourneeResume(BaseModel):
    """Résumé d'une journée de courses."""

    date: date
    nombre_reunions: int
    nombre_courses_total: int
    reunions: List[ReunionSummary] = []

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# RÉPONSES API
# ============================================================================


class RacesTodayResponse(BaseModel):
    """Réponse pour /races/today."""

    date: date
    nombre_courses: int
    courses: List[CourseSummary]

    model_config = ConfigDict(from_attributes=True)


class RaceDetailResponse(BaseModel):
    """Réponse pour /races/{id}."""

    course: CourseDetail

    model_config = ConfigDict(from_attributes=True)


class PaginatedResponse(BaseModel):
    """Réponse paginée générique."""

    total: int
    page: int
    per_page: int
    pages: int
    items: List[CourseSummary]

    model_config = ConfigDict(from_attributes=True)

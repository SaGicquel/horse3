# -*- coding: utf-8 -*-
"""Orchestration end-to-end: collecte → transformation → matching → rapport qualité.
Usage:
    python run_pipeline.py 2025-11-05 2025-11-06
Sans dates => utilise date du jour.
"""

import sys
from datetime import date
from db_connection import get_connection

from collect.pmu_program import collect_program
from collect.pmu_participants import collect_participants
from collect.pmu_performances import collect_performances
from collect.pmu_pools import collect_pools_and_reports
from collect.pmu_market import collect_market_snapshot
from transform.pmu_to_core_fact import (
    transform_program,
    transform_course_level,
    transform_participants,
    transform_rapports,
)
from match.match_horses import run_matching
from report_quality import generate_quality_report

REUNIONS = list(range(1, 21))
COURSES = list(range(1, 16))


def process_date(d: str):
    print(f"=== Traitement date {d} ===")
    collect_program(d)
    collect_participants(d, REUNIONS, COURSES)
    collect_performances(d, REUNIONS, COURSES)
    collect_pools_and_reports(d, REUNIONS, COURSES)
    collect_market_snapshot(d)
    transform_program(d)
    transform_course_level(d)
    transform_participants(d)
    transform_rapports(d)
    run_matching()
    generate_quality_report(d)
    print(f"=== Terminé {d} ===")


if __name__ == "__main__":
    dates = sys.argv[1:] or [date.today().isoformat()]
    for d in dates:
        process_date(d)

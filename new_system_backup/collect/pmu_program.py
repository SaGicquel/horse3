# -*- coding: utf-8 -*-
"""
Collecte programme/reunions/courses PMU vers staging (stg.raw_payloads).
"""
from collect.pmu_common import get_json, store_raw, BASE, FALLBACK_BASE, ddmmyyyy, rate_sleep


def collect_program(date_iso: str):
    d = ddmmyyyy(date_iso)
    # Programme jour (toutes réunions)
    for base in (BASE, FALLBACK_BASE):
        data = get_json(f"{base}/programme/{d}")
        if data:
            store_raw('PMU_API', 'programme_jour', date_iso, data)
            break

    # Par réunion (R1..R20)
    for r in range(1, 21):
        for base in (BASE, FALLBACK_BASE):
            url = f"{base}/programme/{d}/R{r}"
            data = get_json(url)
            if data:
                store_raw('PMU_API', 'programme_reunion', f"{date_iso}|R{r}", data)
                break
        rate_sleep()


if __name__ == '__main__':
    import sys
    from datetime import date
    date_iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    collect_program(date_iso)

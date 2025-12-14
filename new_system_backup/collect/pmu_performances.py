# -*- coding: utf-8 -*-
"""Collecte performances détaillées par course."""
from collect.pmu_common import get_json, store_raw, BASE, FALLBACK_BASE, ddmmyyyy, rate_sleep


def collect_performances(date_iso: str, reunion_nums: list[int], course_nums: list[int]):
    d = ddmmyyyy(date_iso)
    for r in reunion_nums:
        for c in course_nums:
            key = f"{date_iso}|R{r}|C{c}"
            for base in (BASE, FALLBACK_BASE):
                data = get_json(f"{base}/programme/{d}/R{r}/C{c}/performances-detaillees/pretty")
                if data:
                    store_raw('PMU_API','performances', key, data)
                    break
            rate_sleep(0.2)


if __name__ == '__main__':
    import sys
    from datetime import date
    date_iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    collect_performances(date_iso, list(range(1,21)), list(range(1,16)))

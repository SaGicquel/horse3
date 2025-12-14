# -*- coding: utf-8 -*-
"""Collecte participants par course."""
from collect.pmu_common import get_json, store_raw, BASE, FALLBACK_BASE, ddmmyyyy, rate_sleep


def collect_participants(date_iso: str, reunion_nums: list[int], course_nums: list[int]):
    d = ddmmyyyy(date_iso)
    for r in reunion_nums:
        for c in course_nums:
            key = f"{date_iso}|R{r}|C{c}"
            for base in (BASE, FALLBACK_BASE):
                data = get_json(f"{base}/programme/{d}/R{r}/C{c}/participants")
                if data:
                    store_raw('PMU_API','participants', key, data)
                    break
            rate_sleep(0.2)


if __name__ == '__main__':
    import sys, json
    from datetime import date
    date_iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    # Simplification: course nums 1..15
    collect_participants(date_iso, list(range(1,21)), list(range(1,16)))

# -*- coding: utf-8 -*-
"""
Fonctions communes pour collecte PMU: requêtes HTTP et stockage staging.
"""
import requests
import textwrap
import time
from typing import Optional
from db_connection import get_connection
from psycopg2.extras import Json

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/7"
FALLBACK_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7"

UA = "horse3-pipeline/1.0 (+contact: youremail@example.com)"
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.pmu.fr/turf/",
    "Connection": "keep-alive",
}


def get_json(url: str, timeout=15) -> Optional[dict]:
    r = requests.get(url, headers=COMMON_HEADERS, timeout=timeout)
    if r.status_code in (204, 404):
        return None
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    txt = r.text.strip()
    if not txt:
        return None
    if "text/html" in ctype or txt.startswith("<!DOCTYPE html"):
        head = textwrap.shorten(txt.replace("\n", " "), width=160)
        print(f"[WARN] Non-JSON response {r.status_code}: {head}")
        return None
    try:
        return r.json()
    except Exception:
        return None


def store_raw(source: str, endpoint: str, key: str, payload: dict):
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO stg.raw_payloads(source, endpoint, key, payload)
        VALUES (%s, %s, %s, %s::jsonb)
        ON CONFLICT (source, endpoint, key) DO UPDATE SET payload = EXCLUDED.payload, fetched_at = NOW()
        """,
        (source, endpoint, key, Json(payload))
    )
    con.commit()
    con.close()


def ddmmyyyy(date_iso: str) -> str:
    ds = date_iso.replace('-', '')
    y, m, d = ds[:4], ds[4:6], ds[6:8]
    return f"{d}{m}{y}"


def rate_sleep(s: float = 0.25):
    time.sleep(s)

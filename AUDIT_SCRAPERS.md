# 🔍 AUDIT COMPLET DES SCRIPTS DE SCRAPING

**Date de l'audit** : 24 novembre 2025  
**Projet** : Horse3 - Système de scraping PMU

---

## 📋 RÉSUMÉ EXÉCUTIF

| Critère | Note | Commentaire |
|---------|------|-------------|
| **Architecture globale** | ⭐⭐⭐ | Structure présente mais fragmentée |
| **Gestion des erreurs** | ⭐⭐⭐ | Basique, peut être améliorée |
| **Performance** | ⭐⭐⭐⭐ | Multi-threading implémenté |
| **Maintenabilité** | ⭐⭐ | Code dupliqué, manque d'abstraction |
| **Robustesse** | ⭐⭐⭐ | Retry basique, pas de circuit breaker |
| **Logging** | ⭐⭐ | Inconsistant entre les scripts |

---

## 📁 FICHIERS ANALYSÉS

### Scripts Principaux
1. `scraper_pmu_simple.py` (1616 lignes) - **Script principal PMU**
2. `scraper_turfbzh.py` (372 lignes) - **Cotes Turf.bzh**
3. `scraper_zoneturf.py` (542 lignes) - **Données Zone-Turf**
4. `scraper_hippodromes.py` (523 lignes) - **Enrichissement hippodromes**
5. `scraper_dates.py` (266 lignes) - **Multi-dates**
6. `scraper_pmu_adapter.py` (711 lignes) - **Adaptateur schéma**
7. `scraper_zoneturf_html.py` (248 lignes) - **Zone-Turf HTML**
8. `scraper_interactif.py` (210 lignes) - **Interface interactive**
9. `scraper_today.py` (105 lignes) - **Scraping jour courant**
10. `scrape_historique.py` (83 lignes) - **Historique 30 jours**

---

## 🔴 PROBLÈMES CRITIQUES

### 1. Code Dupliqué et Incohérence
**Fichiers concernés** : `scraper_pmu_simple.py`, `scraper_pmu_adapter.py`, `scraper_zoneturf.py`

```python
# Duplication de logique hippodrome dans 3 fichiers différents
# scraper_pmu_simple.py ligne 204
def resolve_hippodrome_identifiers(hippo_data) -> Tuple[Optional[str], Optional[str]]:

# scraper_pmu_adapter.py ligne 62
def resolve_hippodrome_identifiers(self, hippo_data):
```

**Impact** : Maintenance difficile, risque de divergence

**Recommandation** :
```python
# Créer un module utils/scrapers_common.py
class HippodromeResolver:
    @staticmethod
    def resolve(hippo_data) -> Tuple[Optional[str], Optional[str]]:
        """Résout les identifiants d'un hippodrome depuis diverses sources."""
        if not hippo_data:
            return None, None
        # ... logique commune
```

---

### 2. Gestion des Connexions DB Inconsistante
**Fichiers concernés** : Tous les scrapers

```python
# scraper_today.py utilise ENCORE sqlite3 ❌
con = sqlite3.connect(DB_PATH)

# scraper_pmu_simple.py utilise PostgreSQL ✓
from db_connection import get_connection
```

**Impact** : `scraper_today.py` et `scraper_interactif.py` sont CASSÉS (tentent d'utiliser SQLite)

**Recommandation** : Mettre à jour tous les scripts pour utiliser PostgreSQL via `db_connection.py`

---

### 3. Absence de Rate Limiting Intelligent

**Fichier** : `scraper_pmu_simple.py`

```python
RATE_LIMIT_DELAY = 0.1  # Délai très court
time.sleep(0.35)  # Délai fixe dans discover_reunions
```

**Problèmes** :
- Délais fixes non adaptatifs
- Pas de gestion de throttling API
- Risque de blocage par l'API PMU

**Recommandation** :
```python
import time
from functools import wraps

class AdaptiveRateLimiter:
    def __init__(self, min_delay=0.1, max_delay=5.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = min_delay
        self.errors_count = 0
    
    def wait(self):
        time.sleep(self.current_delay)
    
    def success(self):
        self.errors_count = 0
        self.current_delay = max(self.min_delay, self.current_delay * 0.9)
    
    def error(self, status_code=None):
        self.errors_count += 1
        if status_code == 429:  # Too Many Requests
            self.current_delay = min(self.max_delay, self.current_delay * 3)
        else:
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
```

---

### 4. Pas de Circuit Breaker

**Impact** : Si l'API PMU est down, les scripts continuent d'essayer indéfiniment

**Recommandation** :
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF-OPEN
    
    def can_execute(self) -> bool:
        if self.state == 'CLOSED':
            return True
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'HALF-OPEN'
                return True
            return False
        return True  # HALF-OPEN
    
    def record_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

---

## 🟡 PROBLÈMES IMPORTANTS

### 5. Gestion des Sessions HTTP Non Optimisée

**Fichier** : `scraper_pmu_simple.py`

```python
# Création d'une nouvelle requête à chaque appel
def get_json(url: str, timeout=15):
    r = requests.get(url, headers=COMMON_HEADERS, timeout=timeout)
```

**Recommandation** : Utiliser une session persistante
```python
class PMUScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(COMMON_HEADERS)
        # Configurer le pool de connexions
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('https://', adapter)
```

---

### 6. Logging Inconsistant

**Problème** : Mix de `print()` et `logging` selon les fichiers

```python
# scraper_pmu_simple.py - utilise print()
print(f"[WARN] Non-JSON response...")

# scraper_turfbzh.py - utilise logging ✓
logger.warning(f"⚠️  404 - Course non trouvée")
```

**Recommandation** : Unifier avec logging structuré
```python
# utils/logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
        }
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        return json.dumps(log_data)

def setup_logging(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Handler console avec emojis
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    logger.addHandler(console)
    
    # Handler fichier JSON
    file_handler = logging.FileHandler(f'logs/{name}.json')
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    return logger
```

---

### 7. Pas de Validation des Données

**Fichier** : `scraper_pmu_simple.py`

```python
# Les données sont insérées sans validation préalable
cur.execute("""
    INSERT INTO chevaux (nom, date_naissance, sexe, ...)
    VALUES (%s,%s,%s,...)
""", (nom, date_naissance, sexe, ...))
```

**Recommandation** : Ajouter une couche de validation
```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class ChevalData:
    nom: str
    date_naissance: Optional[str] = None
    sexe: Optional[str] = None
    
    def __post_init__(self):
        if not self.nom or len(self.nom) < 2:
            raise ValueError(f"Nom de cheval invalide: {self.nom}")
        if self.sexe and self.sexe not in ('M', 'F', 'H'):
            raise ValueError(f"Sexe invalide: {self.sexe}")
        if self.date_naissance:
            if not re.match(r'\d{4}-\d{2}-\d{2}', self.date_naissance):
                raise ValueError(f"Format date invalide: {self.date_naissance}")

# Utilisation
try:
    cheval = ChevalData(nom=nom, date_naissance=date_naissance, sexe=sexe)
    upsert_cheval(cur, cheval)
except ValueError as e:
    logger.warning(f"Données invalides ignorées: {e}")
```

---

### 8. Thread Safety Incomplet

**Fichier** : `scraper_pmu_simple.py`

```python
db_lock = threading.Lock()

def process_course(args):
    # Le lock est utilisé pour le cursor mais pas pour le commit
    with db_lock:
        cur = con.cursor()
    
    # ... traitement ...
    
    with db_lock:
        con.commit()  # ⚠️ Pas thread-safe si transaction échoue
```

**Recommandation** : Utiliser un pool de connexions
```python
from psycopg2 import pool

class DatabasePool:
    _pool = None
    
    @classmethod
    def get_pool(cls, min_conn=2, max_conn=10):
        if cls._pool is None:
            cls._pool = pool.ThreadedConnectionPool(
                min_conn, max_conn,
                host='localhost',
                port=54624,
                database='pmu_database',
                user='pmu_user',
                password='pmu_secure_password_2025'
            )
        return cls._pool
    
    @classmethod
    def get_connection(cls):
        return cls.get_pool().getconn()
    
    @classmethod
    def return_connection(cls, conn):
        cls.get_pool().putconn(conn)
```

---

## 🟢 POINTS POSITIFS

### ✅ Multi-threading Implémenté
```python
MAX_WORKERS = 8
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_course, task) for task in tasks]
```

### ✅ Gestion Fallback API
```python
for base in (BASE, FALLBACK_BASE):
    data = get_json(f"{base}/programme/{d}/R{reunion}")
    if data:
        break
```

### ✅ Normalisation des Données
```python
def norm(s: str|None) -> str|None:
    """Normalisation nom: strip accents, collapse spaces, lower."""
    
def normalize_race(race: str|None) -> str|None:
    """Normalise un nom de race pour éviter les doublons"""
```

### ✅ Upsert avec COALESCE
```python
ON CONFLICT (nom_norm, race_key) DO UPDATE SET
    id_cheval = COALESCE(EXCLUDED.id_cheval, cheval_courses_seen.id_cheval),
```

---

## 📊 MÉTRIQUES DE PERFORMANCE

### Configuration Actuelle
| Paramètre | Valeur | Recommandation |
|-----------|--------|----------------|
| MAX_WORKERS | 8 | OK (ajuster selon CPU) |
| RATE_LIMIT_DELAY | 0.1s | Augmenter à 0.3s minimum |
| Timeout | 15s | OK |
| Max Retries | 2 | Augmenter à 3 |

### Estimation Temps de Scraping
| Période | Courses estimées | Temps actuel | Temps optimisé |
|---------|------------------|--------------|----------------|
| 1 jour | ~100 courses | ~5 min | ~2 min |
| 7 jours | ~700 courses | ~35 min | ~15 min |
| 30 jours | ~3000 courses | ~2h30 | ~1h |

---

## 🛠️ PLAN D'AMÉLIORATION

### Phase 1 - Corrections Critiques (Urgent) ✅ COMPLÉTÉ
1. ✅ Corriger `scraper_today.py` et `scraper_interactif.py` pour PostgreSQL
2. ✅ Créer module commun `utils/scraper_common.py`
3. ✅ Implémenter rate limiter adaptatif (`utils/rate_limiter.py`)

### Phase 2 - Robustesse (Important) ✅ EN COURS
4. ✅ Ajouter circuit breaker (`utils/circuit_breaker.py`)
5. ⬜ Implémenter pool de connexions DB
6. ⬜ Unifier logging avec rotation des fichiers

### Phase 3 - Optimisation (Souhaitable)
7. ⬜ Ajouter validation des données avec dataclasses
8. ⬜ Implémenter cache Redis pour requêtes répétitives
9. ⬜ Ajouter métriques Prometheus

### Phase 4 - Monitoring (Futur)
10. ⬜ Dashboard Grafana pour suivi scraping
11. ⬜ Alertes Slack/Discord sur erreurs
12. ⬜ Tests de régression automatisés

---

## 📝 RECOMMANDATIONS ARCHITECTURALES

### Structure Proposée

```
horse3/
├── scrapers/
│   ├── __init__.py
│   ├── base.py              # Classe abstraite BaseScraper
│   ├── pmu.py               # PMUScraper hérite de BaseScraper
│   ├── turfbzh.py           # TurfBzhScraper
│   ├── zoneturf.py          # ZoneTurfScraper
│   └── hippodromes.py       # HippodromeScraper
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py      # Rate limiting adaptatif
│   ├── circuit_breaker.py   # Circuit breaker
│   ├── db_pool.py           # Pool connexions PostgreSQL
│   ├── logging_config.py    # Configuration logging
│   └── validators.py        # Validation données
├── models/
│   ├── __init__.py
│   ├── cheval.py            # Dataclass Cheval
│   ├── course.py            # Dataclass Course
│   └── performance.py       # Dataclass Performance
├── orchestrator.py          # Orchestrateur principal
└── cli.py                   # Interface CLI unifiée
```

### Classe de Base Proposée

```python
# scrapers/base.py
from abc import ABC, abstractmethod
import logging
from utils.rate_limiter import AdaptiveRateLimiter
from utils.circuit_breaker import CircuitBreaker
from utils.db_pool import DatabasePool

class BaseScraper(ABC):
    """Classe de base pour tous les scrapers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.rate_limiter = AdaptiveRateLimiter()
        self.circuit_breaker = CircuitBreaker()
        self.session = self._create_session()
        self.stats = {'success': 0, 'errors': 0, 'skipped': 0}
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.get_headers())
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(total=3, backoff_factor=0.5)
        )
        session.mount('https://', adapter)
        return session
    
    @abstractmethod
    def get_headers(self) -> dict:
        """Headers HTTP spécifiques au scraper."""
        pass
    
    @abstractmethod
    def scrape(self, **kwargs):
        """Méthode principale de scraping."""
        pass
    
    def safe_request(self, url: str, **kwargs) -> Optional[dict]:
        """Requête HTTP sécurisée avec rate limiting et circuit breaker."""
        if not self.circuit_breaker.can_execute():
            self.logger.warning(f"Circuit ouvert, requête ignorée: {url}")
            return None
        
        self.rate_limiter.wait()
        
        try:
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            self.circuit_breaker.record_success()
            self.rate_limiter.success()
            return response.json()
        except requests.HTTPError as e:
            self.circuit_breaker.record_failure()
            self.rate_limiter.error(e.response.status_code)
            self.logger.error(f"HTTP {e.response.status_code}: {url}")
            return None
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.rate_limiter.error()
            self.logger.exception(f"Erreur requête: {url}")
            return None
```

---

## 🎯 CONCLUSION

Le système de scraping actuel est fonctionnel mais souffre de plusieurs problèmes de conception qui limitent sa robustesse et sa maintenabilité. Les améliorations prioritaires sont :

1. **Corriger les scripts cassés** (`scraper_today.py`, `scraper_interactif.py`)
2. **Factoriser le code dupliqué** dans un module commun
3. **Implémenter un rate limiting intelligent** pour éviter les blocages API
4. **Unifier le logging** pour faciliter le debugging

L'investissement estimé pour les corrections critiques est de **2-3 jours de développement**.

---

*Audit réalisé par GitHub Copilot - 24/11/2025*

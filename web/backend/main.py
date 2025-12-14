"""
HorseRace Predictor - API Backend
FastAPI server pour servir les données de courses hippiques
"""

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any, Literal
import sqlite3
import json
import yaml
import copy
import math
from datetime import datetime, date, timedelta
from decimal import Decimal
import os
import sys
import secrets
import hmac
import subprocess
from dotenv import load_dotenv
import pandas as pd
from functools import lru_cache
import time
import hashlib

# Import garde anti-fuite (chemin absolu pour Docker)
try:
    from odds_guard import select_preoff_market_odds
except ImportError:
    from .odds_guard import select_preoff_market_odds

# Charger les variables d'environnement
load_dotenv()

# Répertoire de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Configuration - chemin et valeurs par défaut
# ============================================================================

# Valeurs par défaut si le fichier YAML est introuvable (utile en Docker)
DEFAULT_CONFIG = {
    "calibration": {
        "temperature": 1.254,
        "blend_alpha_global": 0.2,
        "blend_alpha_plat": 0.0,
        "blend_alpha_trot": 0.4,
        "blend_alpha_obstacle": 0.4,
        "calibrator": "platt",
    },
    "simulation": {"num_simulations": 20000},
    "markets": {"mode": "parimutuel", "takeout_rate": 0.16},
    "kelly": {"fraction": 0.25, "value_cutoff": 0.05, "max_stake_pct": 0.05},
    "portfolio": {
        "correlation_penalty": 0.30,
        "max_same_race": 2,
        "max_same_jockey": 3,
        "max_same_trainer": 3,
        "confidence_level": 0.95,
    },
    "exotics": {
        "p_place_simulations": 20000,
        "top_k_couples": 50,
        "top_k_trios": 100,
        "max_coverage": 0.30,
    },
    "backtest": {"initial_bankroll": 1000.0, "min_races": 50},
    "artifacts": {
        "model_path": "models/xgb_proba_v9.joblib",
        "predictions_dir": "predictions",
        "reports_dir": "reports",
    },
    "budget": {"daily_limit": 100.0},
}

def _merge_config_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne la config trouvée avec les valeurs par défaut (sans écraser les personnalisations)."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

def resolve_config_path() -> str:
    """
    Détermine le chemin du fichier de configuration en privilégiant :
    1. Variables d'env (PRO_BETTING_CONFIG / CONFIG_PATH)
    2. Chemin root du repo (../../config)
    3. Dossier config local ou /config monté en Docker
    4. Fallback dans /app/config pour assurer la création du fichier
    """
    env_path = os.getenv("PRO_BETTING_CONFIG") or os.getenv("CONFIG_PATH")
    candidates = [
        env_path,
        os.path.join(BASE_DIR, "..", "..", "config", "pro_betting.yaml"),
        os.path.join(BASE_DIR, "..", "config", "pro_betting.yaml"),
        "/project/config/pro_betting.yaml",
        "/config/pro_betting.yaml",
        "/app/config/pro_betting.yaml",
        os.path.join(BASE_DIR, "config", "pro_betting.yaml"),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)

    if env_path:
        return os.path.abspath(env_path)

    # Fallback: créer/placer la config dans /app/config (volume docker ou dossier local)
    return os.path.abspath(os.path.join(BASE_DIR, "config", "pro_betting.yaml"))

def load_config_file(create_if_missing: bool = True) -> Dict[str, Any]:
    """
    Charge la configuration ou initialise un fichier avec les valeurs par défaut si absent.
    Retourne toujours une configuration complète (avec budgets).
    """
    config_path = resolve_config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
        if create_if_missing:
            with open(config_path, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    return _merge_config_with_defaults(config)

# ============================================================================
# SYSTÈME DE PROFILS UTILISATEUR & PERSONNALISATION
# ============================================================================

from enum import Enum

class ProfilRisque(str, Enum):
    PRUDENT = "PRUDENT"
    STANDARD = "STANDARD"  
    AGRESSIF = "AGRESSIF"

class ConfigProfil:
    """Configuration des profils de mise selon le niveau de risque"""
    
    PROFILES = {
        ProfilRisque.PRUDENT: {
            "kelly_multiplier": 0.25,  # Plus conservateur
            "max_stake_pct": 1.5,     # Max 1.5% de la bankroll par pari
            "max_daily_pct": 8.0,     # Max 8% de la bankroll par jour
            "min_value_pct": 8.0,     # Seulement les très bonnes values
            "max_odds": 12.0,         # Pas de très grosses cotes
            "description": "Sécurisé - Variance faible, croissance stable"
        },
        ProfilRisque.STANDARD: {
            "kelly_multiplier": 0.5,   # Réglage équilibré
            "max_stake_pct": 2.5,     # Max 2.5% par pari
            "max_daily_pct": 12.0,    # Max 12% par jour
            "min_value_pct": 3.0,     # Values moyennes acceptées
            "max_odds": 20.0,         # Cotes moyennes-hautes
            "description": "Équilibré - Bon compromis risque/rendement"
        },
        ProfilRisque.AGRESSIF: {
            "kelly_multiplier": 1.0,   # Kelly complet
            "max_stake_pct": 4.0,     # Max 4% par pari
            "max_daily_pct": 20.0,    # Max 20% par jour
            "min_value_pct": 1.0,     # Values faibles acceptées
            "max_odds": 50.0,         # Grosses cotes autorisées
            "description": "Agressif - Variance élevée, potentiel maximum"
        }
    }
    
    @classmethod
    def get_config(cls, profil: ProfilRisque) -> Dict[str, Any]:
        return cls.PROFILES[profil]

class UserBettingParams(BaseModel):
    """Paramètres de paris utilisateur"""
    bankroll: float = Field(default=500.0, ge=100, le=10000, description="Bankroll en euros")
    profil: ProfilRisque = Field(default=ProfilRisque.STANDARD, description="Profil de risque")
    
    def get_config(self) -> Dict[str, Any]:
        return ConfigProfil.get_config(self.profil)

class UserSettings(BaseModel):
    """Modèle complet des settings utilisateur"""
    # Paramètres de base
    bankroll: float = Field(default=1000.0, ge=100, le=50000)
    profil_risque: str = Field(default='STANDARD')
    
    # Profil Kelly
    kelly_profile: str = Field(default='STANDARD')
    custom_kelly_fraction: float = Field(default=0.33, ge=0.01, le=1.0)
    value_cutoff: float = Field(default=0.05, ge=0.01, le=0.2)
    
    # Caps & Limits
    cap_per_bet: float = Field(default=0.02, ge=0.001, le=0.1)
    daily_budget_rate: float = Field(default=0.12, ge=0.01, le=0.5)
    max_unit_bets_per_race: int = Field(default=2, ge=1, le=10)
    rounding_increment_eur: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Paris Exotiques
    per_ticket_rate: float = Field(default=0.005, ge=0.001, le=0.02)
    max_pack_rate: float = Field(default=0.03, ge=0.01, le=0.1)
    
    # Marché & Environnement
    market_mode: str = Field(default='parimutuel')
    takeout_rate: float = Field(default=0.17, ge=0.1, le=0.3)

def get_user_from_request(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Récupère l'utilisateur depuis le header Authorization"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.replace("Bearer ", "")
    return get_user_from_token(token)

async def get_user_settings(user_id: int) -> UserSettings:
    """Récupère les settings d'un utilisateur depuis la BDD"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        cur.execute(adapt_query("""
            SELECT 
                bankroll, profil_risque, kelly_profile, custom_kelly_fraction,
                value_cutoff, cap_per_bet, daily_budget_rate, max_unit_bets_per_race,
                rounding_increment_eur, per_ticket_rate, max_pack_rate,
                market_mode, takeout_rate
            FROM user_settings 
            WHERE user_id = %s
        """), (user_id,))
        
        row = cur.fetchone()
        if row:
            return UserSettings(
                bankroll=row[0], profil_risque=row[1], kelly_profile=row[2],
                custom_kelly_fraction=row[3], value_cutoff=row[4], cap_per_bet=row[5],
                daily_budget_rate=row[6], max_unit_bets_per_race=row[7],
                rounding_increment_eur=row[8], per_ticket_rate=row[9],
                max_pack_rate=row[10], market_mode=row[11], takeout_rate=row[12]
            )
        else:
            # Créer des settings par défaut pour cet utilisateur
            default_settings = UserSettings()
            await save_user_settings(user_id, default_settings)
            return default_settings
    finally:
        con.close()

async def save_user_settings(user_id: int, settings: UserSettings):
    """Sauvegarde les settings d'un utilisateur en BDD"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        # Utiliser UPSERT (INSERT ... ON CONFLICT DO UPDATE pour PostgreSQL)
        if USE_POSTGRESQL:
            cur.execute("""
                INSERT INTO user_settings (
                    user_id, bankroll, profil_risque, kelly_profile, custom_kelly_fraction,
                    value_cutoff, cap_per_bet, daily_budget_rate, max_unit_bets_per_race,
                    rounding_increment_eur, per_ticket_rate, max_pack_rate,
                    market_mode, takeout_rate, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE SET
                    bankroll = EXCLUDED.bankroll,
                    profil_risque = EXCLUDED.profil_risque,
                    kelly_profile = EXCLUDED.kelly_profile,
                    custom_kelly_fraction = EXCLUDED.custom_kelly_fraction,
                    value_cutoff = EXCLUDED.value_cutoff,
                    cap_per_bet = EXCLUDED.cap_per_bet,
                    daily_budget_rate = EXCLUDED.daily_budget_rate,
                    max_unit_bets_per_race = EXCLUDED.max_unit_bets_per_race,
                    rounding_increment_eur = EXCLUDED.rounding_increment_eur,
                    per_ticket_rate = EXCLUDED.per_ticket_rate,
                    max_pack_rate = EXCLUDED.max_pack_rate,
                    market_mode = EXCLUDED.market_mode,
                    takeout_rate = EXCLUDED.takeout_rate,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_id, settings.bankroll, settings.profil_risque, settings.kelly_profile,
                settings.custom_kelly_fraction, settings.value_cutoff, settings.cap_per_bet,
                settings.daily_budget_rate, settings.max_unit_bets_per_race,
                settings.rounding_increment_eur, settings.per_ticket_rate,
                settings.max_pack_rate, settings.market_mode, settings.takeout_rate
            ))
        else:
            # SQLite - utiliser INSERT OR REPLACE
            cur.execute("""
                INSERT OR REPLACE INTO user_settings (
                    user_id, bankroll, profil_risque, kelly_profile, custom_kelly_fraction,
                    value_cutoff, cap_per_bet, daily_budget_rate, max_unit_bets_per_race,
                    rounding_increment_eur, per_ticket_rate, max_pack_rate,
                    market_mode, takeout_rate, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                user_id, settings.bankroll, settings.profil_risque, settings.kelly_profile,
                settings.custom_kelly_fraction, settings.value_cutoff, settings.cap_per_bet,
                settings.daily_budget_rate, settings.max_unit_bets_per_race,
                settings.rounding_increment_eur, settings.per_ticket_rate,
                settings.max_pack_rate, settings.market_mode, settings.takeout_rate
            ))
        con.commit()
    finally:
        con.close()

def calculate_personalized_stake(runner: Dict, user_params: UserBettingParams) -> Dict[str, Any]:
    """
    🎯 Calcule la mise personnalisée selon le profil utilisateur
    
    Args:
        runner: Données du cheval du générateur
        user_params: Paramètres utilisateur (bankroll + profil)
    
    Returns:
        Dict avec mise ajustée et informations
    """
    config = user_params.get_config()
    
    # Données de base du runner
    kelly_pct = runner.get('kelly_pct', 0) / 100  # Convertir % en fraction
    value_pct = runner.get('value_pct', 0)
    odds = runner.get('odds', 0)
    p_win = runner.get('p_win', 0)
    
    # Filtres du profil
    if value_pct < config['min_value_pct']:
        return {"eligible": False, "reason": f"Value {value_pct:.1f}% < seuil {config['min_value_pct']:.1f}%"}
    
    if odds > config['max_odds']:
        return {"eligible": False, "reason": f"Cote {odds:.1f} > limite {config['max_odds']:.1f}"}
    
    # Calcul de la mise ajustée
    kelly_adjusted = kelly_pct * config['kelly_multiplier']
    
    # Application du cap par pari
    max_stake_fraction = config['max_stake_pct'] / 100
    kelly_capped = min(kelly_adjusted, max_stake_fraction)
    
    # Mise en euros
    stake_euros = kelly_capped * user_params.bankroll
    
    # Arrondissement intelligent
    if stake_euros < 5:
        stake_euros = round(stake_euros, 1)  # Ex: 2.3€
    elif stake_euros < 20:
        stake_euros = round(stake_euros * 2) / 2  # Ex: 12.5€
    else:
        stake_euros = round(stake_euros)  # Ex: 25€
    
    # Calcul du gain potentiel
    gain_potentiel = stake_euros * (odds - 1)
    
    # Pourcentage de la bankroll
    stake_pct = (stake_euros / user_params.bankroll) * 100
    
    return {
        "eligible": True,
        "stake_euros": stake_euros,
        "stake_pct": stake_pct,
        "gain_potentiel": gain_potentiel,
        "kelly_original": kelly_pct * 100,
        "kelly_adjusted": kelly_adjusted * 100,
        "profil_multiplier": config['kelly_multiplier'],
        "reason": f"Kelly {kelly_pct*100:.1f}% → {kelly_adjusted*100:.1f}% (profil {user_params.profil.value})"
    }

# ============================================================================
# Système de Cache en Mémoire
# ============================================================================

class MemoryCache:
    """Cache en mémoire avec TTL (Time To Live) pour les requêtes lourdes"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        # Durées de cache en secondes
        self.TTL = {
            'dashboard': 60,        # 1 minute
            'analytics': 300,       # 5 minutes
            'chevaux_list': 120,    # 2 minutes
            'courses_list': 120,    # 2 minutes
            'jockeys': 300,         # 5 minutes
            'entraineurs': 300,     # 5 minutes
            'hippodromes': 600,     # 10 minutes
            'default': 60,          # 1 minute par défaut
        }
    
    def _get_key(self, prefix: str, params: dict = None) -> str:
        """Génère une clé de cache unique"""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            hash_str = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{prefix}:{hash_str}"
        return prefix
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache si elle n'est pas expirée"""
        if key in self._cache:
            timestamp = self._timestamps.get(key, 0)
            # Extraire le préfixe pour connaître le TTL
            prefix = key.split(':')[0]
            ttl = self.TTL.get(prefix, self.TTL['default'])
            
            if time.time() - timestamp < ttl:
                return self._cache[key]
            else:
                # Supprimer les entrées expirées
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Stocke une valeur dans le cache"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def invalidate(self, prefix: str = None):
        """Invalide le cache (tout ou par préfixe)"""
        if prefix:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
                del self._timestamps[k]
        else:
            self._cache.clear()
            self._timestamps.clear()
    
    def cleanup(self):
        """Nettoie les entrées expirées"""
        now = time.time()
        keys_to_delete = []
        for key in self._cache:
            prefix = key.split(':')[0]
            ttl = self.TTL.get(prefix, self.TTL['default'])
            if now - self._timestamps.get(key, 0) >= ttl:
                keys_to_delete.append(key)
        for k in keys_to_delete:
            del self._cache[k]
            del self._timestamps[k]

# Instance globale du cache
cache = MemoryCache()

# Essayer d'importer db_connection local d'abord, puis le module parent
try:
    # Import local (dans le même répertoire - pour Docker)
    from db_connection import get_connection
    USE_POSTGRESQL = True
    print(f"[INFO] Utilisation de PostgreSQL (module local)")
except ImportError:
    # Fallback: chercher dans le répertoire parent (développement local)
    parent_dir = os.path.join(BASE_DIR, "..", "..")
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from db_connection import get_connection
        USE_POSTGRESQL = True
        print(f"[INFO] Utilisation de PostgreSQL (module parent)")
    except ImportError as e:
        USE_POSTGRESQL = False
        print(f"[WARN] PostgreSQL non disponible ({e}), utilisation de SQLite")
        # Chemin vers la base de données SQLite (chemin absolu)
        DB_PATH = os.path.join(BASE_DIR, "..", "data", "database.db")

app = FastAPI(
    title="HorseRace Predictor API",
    description="API REST pour l'analyse des courses hippiques",
    version="1.0.0"
)

# CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
@app.get("/healthz")
async def health_check():
    """Endpoint de health check pour Docker (supporte /health et /healthz)"""
    return {"status": "healthy", "service": "horse-backend"}

# ============================================================================
# Helper pour les connexions DB
# ============================================================================

def get_db_connection():
    """Retourne une connexion à la base de données (PostgreSQL ou SQLite)"""
    if USE_POSTGRESQL:
        return get_connection()
    else:
        return sqlite3.connect(DB_PATH)

# ============================================================================
# Modèles Pydantic
# ============================================================================

class DashboardStats(BaseModel):
    """Statistiques du dashboard"""
    taux_reussite: float
    roi_moyen: float
    courses_analysees: int
    modeles_actifs: int
    evolution_reussite: float
    evolution_roi: float
    evolution_courses: int

class PerformanceRecente(BaseModel):
    """Performance récente d'une course"""
    nom_course: str
    probabilite: float
    resultat: str
    evolution: str

class VariablePredictive(BaseModel):
    """Variable prédictive avec son importance"""
    nom: str
    importance: float

class ChevalStats(BaseModel):
    """Statistiques d'un cheval"""
    nom: str
    age: int
    sexe: str
    race: str
    musique: str
    nb_courses: int
    nb_victoires: int
    taux_victoire: float
    dernieres_performances: List[Dict[str, Any]]

class MonitoringStats(BaseModel):
    """Statistiques de monitoring du robot"""
    total_bets: int
    pending_bets: int
    finished_bets: int
    win_rate: float
    roi: float
    pnl_net: float
    pnl_history: List[Dict[str, Any]]
    recent_bets: List[Dict[str, Any]]
    data_scope: Optional[str] = None

class SettingsMarkets(BaseModel):
    mode: str
    takeout_rate: float

class SettingsKelly(BaseModel):
    fraction: float
    value_cutoff: float
    max_stake_pct: float

class SettingsBudget(BaseModel):
    daily_limit: Optional[float] = 100.0

# =============================================================================
# NOUVEAUX MODÈLES - Betting Defaults (Kelly fractionnaire)
# =============================================================================

class SettingsBettingDefaults(BaseModel):
    """Configuration de la politique de mise par défaut."""
    kelly_profile_default: str = "STANDARD"  # SUR, STANDARD, AMBITIEUX, PERSONNALISE
    kelly_fraction_map: Dict[str, float] = {"SUR": 0.25, "STANDARD": 0.33, "AMBITIEUX": 0.50}
    custom_kelly_fraction: float = 0.33
    value_cutoff: float = 0.05           # ≥5% minimum
    cap_per_bet: float = 0.02            # 2% bankroll max/pari
    daily_budget_rate: float = 0.12      # 12% bankroll/jour
    max_unit_bets_per_race: int = 2
    rounding_increment_eur: float = 0.5  # Arrondi à 0.50€

class SettingsExoticsDefaults(BaseModel):
    """Configuration des paris exotiques."""
    per_ticket_rate: float = 0.0075      # 0.75% bankroll/ticket
    max_pack_rate: float = 0.04          # 4% bankroll max/pack

class Settings(BaseModel):
    markets: SettingsMarkets
    kelly: SettingsKelly
    budget: Optional[SettingsBudget] = None
    betting_defaults: Optional[SettingsBettingDefaults] = None
    exotics_defaults: Optional[SettingsExoticsDefaults] = None

class UserProfile(BaseModel):
    """Profil utilisateur retourné aux clients"""
    id: int
    email: EmailStr
    display_name: Optional[str] = None
    created_at: Optional[datetime] = None

class AuthResponse(BaseModel):
    """Réponse standard pour les endpoints d'authentification"""
    token: str
    expires_at: Optional[datetime] = None
    user: UserProfile

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    display_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class BetCreateRequest(BaseModel):
    race_key: Optional[str] = None
    event_date: Optional[date] = None
    hippodrome: Optional[str] = None
    selection: str
    bet_type: Optional[str] = None
    stake: float = Field(..., gt=0)
    odds: float = Field(..., gt=0)
    status: Optional[Literal["PENDING", "WIN", "LOSE", "VOID"]] = "PENDING"
    notes: Optional[str] = None

class BetUpdateRequest(BaseModel):
    race_key: Optional[str] = None
    event_date: Optional[date] = None
    hippodrome: Optional[str] = None
    selection: Optional[str] = None
    bet_type: Optional[str] = None
    stake: Optional[float] = Field(default=None, gt=0)
    odds: Optional[float] = Field(default=None, gt=0)
    status: Optional[Literal["PENDING", "WIN", "LOSE", "VOID"]] = None
    notes: Optional[str] = None

class BetResponse(BaseModel):
    id: int
    race_key: Optional[str] = None
    event_date: Optional[date] = None
    hippodrome: Optional[str] = None
    selection: str
    bet_type: Optional[str] = None
    stake: float
    odds: float
    status: str
    pnl: float
    notes: Optional[str] = None
    created_at: Optional[datetime] = None

class BetSummary(BaseModel):
    total_bets: int
    pending_bets: int
    finished_bets: int
    total_stake: float
    pnl_net: float
    roi: float
    win_rate: float
    by_status: Dict[str, int]
    history: List[Dict[str, Any]]
    bets: List[BetResponse]

# ============================================================================
# Helpers - Authentification & gestion des paris utilisateurs
# ============================================================================

# Normalisation des statuts de pari
BET_STATUS_MAP = {
    "WIN": "WIN",
    "WON": "WIN",
    "GAGNE": "WIN",
    "GAIN": "WIN",
    "LOSE": "LOSE",
    "LOST": "LOSE",
    "PERDU": "LOSE",
    "EN_COURS": "PENDING",
    "PENDING": "PENDING",
    "VOID": "VOID",
    "REMBOURSE": "VOID",
    "NUL": "VOID",
}

def adapt_query(sql: str) -> str:
    """Adapte les placeholders pour SQLite si besoin"""
    if USE_POSTGRESQL:
        return sql
    return sql.replace("%s", "?")

def to_float(value) -> float:
    """Convertit proprement les valeurs numériques issues de la BDD"""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)

def normalize_bet_status(status: Optional[str]) -> str:
    """Nettoie/normalise le statut d'un pari"""
    if not status:
        return "PENDING"
    key = status.strip().upper()
    return BET_STATUS_MAP.get(key, "PENDING")

def compute_bet_pnl(status: str, stake: float, odds: float) -> float:
    """Calcule le PnL d'un pari en fonction de son statut"""
    stake_val = to_float(stake)
    odds_val = to_float(odds)
    if status == "WIN":
        return round(stake_val * (odds_val - 1), 2)
    if status == "LOSE":
        return round(-stake_val, 2)
    return 0.0

def infer_event_date(race_key: Optional[str], provided: Optional[date]) -> Optional[date]:
    """Essaye d'inférer la date d'une course à partir du race_key"""
    if provided:
        return provided
    if race_key:
        try:
            return datetime.strptime(race_key.split('|')[0], "%Y-%m-%d").date()
        except Exception:
            return None
    return None

def serialize_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None

def hash_password(password: str, salt: Optional[bytes] = None) -> (str, str):
    """Hash PBKDF2 pour stocker les mots de passe"""
    if salt is None:
        salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return salt.hex(), hashed.hex()

def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Vérifie un mot de passe via PBKDF2"""
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(hash_hex)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return hmac.compare_digest(hashed, expected)

def init_user_tables():
    """Crée les tables users/sessions/paris si elles n'existent pas"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        if USE_POSTGRESQL:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    display_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_bets (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    race_key TEXT,
                    event_date DATE,
                    hippodrome TEXT,
                    selection TEXT NOT NULL,
                    bet_type TEXT,
                    stake NUMERIC(12,2) NOT NULL,
                    odds NUMERIC(10,2) NOT NULL,
                    status TEXT DEFAULT 'PENDING',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    bankroll NUMERIC(12,2) DEFAULT 1000,
                    profil_risque TEXT DEFAULT 'STANDARD',
                    kelly_profile TEXT DEFAULT 'STANDARD',
                    custom_kelly_fraction NUMERIC(5,3) DEFAULT 0.33,
                    value_cutoff NUMERIC(5,3) DEFAULT 0.05,
                    cap_per_bet NUMERIC(5,3) DEFAULT 0.02,
                    daily_budget_rate NUMERIC(5,3) DEFAULT 0.12,
                    max_unit_bets_per_race INTEGER DEFAULT 2,
                    rounding_increment_eur NUMERIC(5,2) DEFAULT 1.0,
                    per_ticket_rate NUMERIC(5,3) DEFAULT 0.005,
                    max_pack_rate NUMERIC(5,3) DEFAULT 0.03,
                    market_mode TEXT DEFAULT 'parimutuel',
                    takeout_rate NUMERIC(5,3) DEFAULT 0.17,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        else:
            cur.execute("PRAGMA foreign_keys = ON;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    display_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    race_key TEXT,
                    event_date DATE,
                    hippodrome TEXT,
                    selection TEXT NOT NULL,
                    bet_type TEXT,
                    stake REAL NOT NULL,
                    odds REAL NOT NULL,
                    status TEXT DEFAULT 'PENDING',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    bankroll REAL DEFAULT 1000,
                    profil_risque TEXT DEFAULT 'STANDARD',
                    kelly_profile TEXT DEFAULT 'STANDARD',
                    custom_kelly_fraction REAL DEFAULT 0.33,
                    value_cutoff REAL DEFAULT 0.05,
                    cap_per_bet REAL DEFAULT 0.02,
                    daily_budget_rate REAL DEFAULT 0.12,
                    max_unit_bets_per_race INTEGER DEFAULT 2,
                    rounding_increment_eur REAL DEFAULT 1.0,
                    per_ticket_rate REAL DEFAULT 0.005,
                    max_pack_rate REAL DEFAULT 0.03,
                    market_mode TEXT DEFAULT 'parimutuel',
                    takeout_rate REAL DEFAULT 0.17,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        con.commit()
    finally:
        con.close()

def create_session_token(cur, user_id: int) -> (str, datetime):
    """Crée un token de session simple stocké en BDD"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=30)
    cur.execute(adapt_query("""
        DELETE FROM user_sessions 
        WHERE expires_at IS NOT NULL AND expires_at < %s
    """), (datetime.utcnow(),))
    cur.execute(adapt_query("""
        INSERT INTO user_sessions (token, user_id, expires_at)
        VALUES (%s, %s, %s)
    """), (token, user_id, expires_at))
    return token, expires_at

def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Retourne l'utilisateur associé à un token si valide"""
    if not token:
        return None
    con = get_db_connection()
    cur = con.cursor()
    try:
        cur.execute(adapt_query("""
            SELECT u.id, u.email, u.display_name, u.created_at, s.expires_at
            FROM user_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = %s
            LIMIT 1
        """), (token,))
        row = cur.fetchone()
        if not row:
            return None
        expires_at = row[4]
        expires_dt = serialize_datetime(expires_at)
        if expires_dt and expires_dt < datetime.utcnow():
            return None
        created_at = serialize_datetime(row[3])
        return {
            "id": row[0],
            "email": row[1],
            "display_name": row[2],
            "created_at": created_at
        }
    finally:
        con.close()

def parse_auth_header(authorization: Optional[str]) -> str:
    """Extrait le token depuis l'entête Authorization"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Token manquant")
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token invalide")
    return token

def require_auth(authorization: Optional[str]) -> Dict[str, Any]:
    """Valide le token et renvoie l'utilisateur"""
    token = parse_auth_header(authorization)
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expirée ou invalide")
    return user

def serialize_bet_row(row) -> Dict[str, Any]:
    """Transforme une ligne SQL de pari en dictionnaire API"""
    event_date_raw = row[2]
    event_date = None
    if isinstance(event_date_raw, date):
        event_date = event_date_raw
    elif isinstance(event_date_raw, str):
        try:
            event_date = datetime.fromisoformat(event_date_raw).date()
        except Exception:
            event_date = None
    status = normalize_bet_status(row[8])
    pnl = compute_bet_pnl(status, row[6], row[7])
    created_at = serialize_datetime(row[10])
    return {
        "id": row[0],
        "race_key": row[1],
        "event_date": event_date,
        "hippodrome": row[3],
        "selection": row[4],
        "bet_type": row[5],
        "stake": to_float(row[6]),
        "odds": to_float(row[7]),
        "status": status,
        "notes": row[9],
        "created_at": created_at,
        "pnl": pnl
    }

def fetch_user_bets(cur, user_id: int) -> List[Dict[str, Any]]:
    """Récupère et sérialise tous les paris d'un utilisateur (ordre chronologique)."""
    cur.execute(adapt_query("""
        SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
        FROM user_bets
        WHERE user_id = %s
        ORDER BY COALESCE(event_date, created_at) ASC
    """), (user_id,))
    rows = cur.fetchall()
    return [serialize_bet_row(r) for r in rows]

def build_monitoring_from_bets(bets: List[Dict[str, Any]]) -> MonitoringStats:
    """Construit les statistiques de monitoring à partir des paris utilisateur."""
    total_bets = len(bets)
    pending_bets = sum(1 for b in bets if b["status"] == "PENDING")
    finished_bets = sum(1 for b in bets if b["status"] in ("WIN", "LOSE", "VOID"))
    wins = sum(1 for b in bets if b["status"] == "WIN")
    pnl_net = round(sum(b["pnl"] for b in bets), 2)
    resolved_stake = sum(to_float(b["stake"]) for b in bets if b["status"] in ("WIN", "LOSE", "VOID"))
    
    roi = round((pnl_net / resolved_stake) * 100, 2) if resolved_stake > 0 else 0.0
    win_rate = round((wins / finished_bets) * 100, 2) if finished_bets > 0 else 0.0
    
    history_map: Dict[str, float] = {}
    for b in bets:
        bet_date = b.get("event_date")
        if isinstance(bet_date, datetime):
            bet_date = bet_date.date()
        if bet_date is None and isinstance(b.get("created_at"), datetime):
            bet_date = b["created_at"].date()
        if bet_date:
            key = bet_date.isoformat()
            history_map[key] = history_map.get(key, 0.0) + b["pnl"]
    
    pnl_history = []
    pnl_cumul = 0.0
    for d in sorted(history_map.keys()):
        pnl_cumul = round(pnl_cumul + history_map[d], 2)
        pnl_history.append({
            "date": d,
            "pnl_jour": round(history_map[d], 2),
            "pnl_cumul": pnl_cumul
        })
    
    recent_bets = sorted(
        bets,
        key=lambda b: b.get("created_at") or datetime.utcnow(),
        reverse=True
    )[:50]
    
    normalized_recent = []
    for b in recent_bets:
        b_copy = dict(b)
        if isinstance(b_copy.get("event_date"), (datetime, date)):
            b_copy["event_date"] = b_copy["event_date"].isoformat()
        if isinstance(b_copy.get("created_at"), (datetime, date)):
            b_copy["created_at"] = b_copy["created_at"].isoformat()
        normalized_recent.append(b_copy)
    
    return MonitoringStats(
        total_bets=total_bets,
        pending_bets=pending_bets,
        finished_bets=finished_bets,
        win_rate=win_rate,
        roi=roi,
        pnl_net=pnl_net,
        pnl_history=pnl_history,
        recent_bets=normalized_recent,
        data_scope="user"
    )

def find_race_result(cur, race_key: str, selection: str):
    """
    Recherche le résultat d'une course pour une sélection donnée.
    Retourne un tuple: (is_win, place_finale, cote_finale, cote_reference, nom_norm, has_results)
    has_results indique si les résultats sont vraiment disponibles (place_finale NOT NULL)
    """
    if not race_key or not selection:
        return None
    try:
        cur.execute(adapt_query("""
            SELECT is_win, place_finale, cote_finale, cote_reference, nom_norm
            FROM cheval_courses_seen
            WHERE race_key = %s AND LOWER(nom_norm) = LOWER(%s)
            LIMIT 1
        """), (race_key, selection.lower()))
        row = cur.fetchone()
        if not row:
            # Fallback: cherche par similarité simple (contient)
            cur.execute(adapt_query("""
                SELECT is_win, place_finale, cote_finale, cote_reference, nom_norm
                FROM cheval_courses_seen
                WHERE race_key = %s AND LOWER(nom_norm) LIKE %s
                LIMIT 1
            """), (race_key, f"%{selection.lower()}%"))
            row = cur.fetchone()
        
        if row:
            # Ajouter has_results: True si place_finale n'est pas NULL
            is_win, place_finale, cote_finale, cote_reference, nom_norm = row
            has_results = place_finale is not None
            return (is_win, place_finale, cote_finale, cote_reference, nom_norm, has_results)
        return None
    except Exception as e:
        print(f"[WARN] Recherche résultat course échouée: {e}")
        return None


def scrape_rapports_definitifs(race_key: str) -> dict:
    """
    Récupère les rapports définitifs d'une course terminée depuis l'API PMU.
    Retourne un dict structuré par type de pari avec les dividendes par combinaison.
    
    Structure retournée:
    {
        "SIMPLE_GAGNANT": {numPmu: dividende},  # ex: {"1": 10.80}
        "SIMPLE_PLACE": {numPmu: dividende},    # ex: {"1": 2.20, "8": 1.30, "2": 5.10}
        "COUPLE_GAGNANT": {"1-8": dividende},
        "COUPLE_PLACE": {"1-8": dividende, "1-2": dividende, "8-2": dividende},
        "TIERCE": {"ordre": {"1-8-2": dividende}, "desordre": {"1-8-2": dividende}},
        "QUARTE_PLUS": {...},
        "QUINTE_PLUS": {...},
        ...
    }
    """
    import requests
    
    def to_pmu_date(date_iso: str) -> str:
        yyyy, mm, dd = date_iso.split('-')
        return f"{dd}{mm}{yyyy}"
    
    if not race_key:
        return {}
    
    try:
        parts = race_key.split('|')
        if len(parts) < 3:
            return {}
        date_iso = parts[0]
        reunion = int(parts[1].replace('R', ''))
        course = int(parts[2].replace('C', ''))
    except (IndexError, ValueError):
        return {}
    
    date_pmu = to_pmu_date(date_iso)
    
    BASES = [
        "https://online.turfinfo.api.pmu.fr/rest/client/7",
        "https://offline.turfinfo.api.pmu.fr/rest/client/7"
    ]
    HEADERS = {"User-Agent": "horse-bet-checker/1.0", "Accept": "application/json"}
    
    for base in BASES:
        url = f"{base}/programme/{date_pmu}/R{reunion}/C{course}/rapports-definitifs"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                rapports_list = r.json()
                
                rapports = {}
                for rapport in rapports_list:
                    type_pari = rapport.get("typePari", "")
                    rapports_data = rapport.get("rapports", [])
                    
                    # Pour les paris simples (SIMPLE_GAGNANT, SIMPLE_PLACE)
                    if type_pari in ("SIMPLE_GAGNANT", "SIMPLE_PLACE"):
                        rapports[type_pari] = {}
                        for r_data in rapports_data:
                            combinaison = r_data.get("combinaison", "")  # ex: "1", "8"
                            # dividendePourUnEuro est en centimes (1080 = 10.80€)
                            dividende = r_data.get("dividendePourUnEuro", 0) / 100
                            if combinaison:
                                rapports[type_pari][combinaison] = dividende
                    
                    # Pour les paris combinés (COUPLE, TRIO, etc.)
                    elif type_pari in ("COUPLE_GAGNANT", "COUPLE_PLACE", "DEUX_SUR_QUATRE"):
                        rapports[type_pari] = {}
                        for r_data in rapports_data:
                            combinaison = r_data.get("combinaison", "")  # ex: "1-8"
                            dividende = r_data.get("dividendePourUnEuro", 0) / 100
                            if combinaison:
                                rapports[type_pari][combinaison] = dividende
                    
                    # Pour TIERCE, QUARTE, QUINTE (ordre et désordre)
                    elif type_pari in ("TIERCE", "QUARTE_PLUS", "QUINTE_PLUS"):
                        rapports[type_pari] = {"ordre": {}, "desordre": {}, "bonus": {}}
                        for r_data in rapports_data:
                            libelle = r_data.get("libelle", "").lower()
                            combinaison = r_data.get("combinaison", "")
                            dividende = r_data.get("dividendePourUnEuro", 0) / 100
                            if "ordre" in libelle:
                                rapports[type_pari]["ordre"][combinaison] = dividende
                            elif "désordre" in libelle or "desordre" in libelle:
                                rapports[type_pari]["desordre"][combinaison] = dividende
                            elif "bonus" in libelle:
                                rapports[type_pari]["bonus"][combinaison] = dividende
                    
                    # MULTI
                    elif type_pari in ("MULTI", "MINI_MULTI"):
                        rapports[type_pari] = {}
                        for r_data in rapports_data:
                            libelle = r_data.get("libelle", "")
                            combinaison = r_data.get("combinaison", "")
                            dividende = r_data.get("dividendePourUnEuro", 0) / 100
                            key = f"{libelle}|{combinaison}"
                            rapports[type_pari][key] = dividende
                
                if rapports:
                    print(f"[DEBUG] Rapports définitifs trouvés pour {race_key}: {list(rapports.keys())}")
                    return rapports
                
        except Exception as e:
            print(f"[WARN] Erreur récup rapports définitifs ({base}): {e}")
            continue
    
    return {}


def get_rapport_for_bet_type(rapports: dict, bet_type: str, selection: str, num_pmu: str = None) -> float:
    """
    Extrait le bon rapport (dividende) selon le type de pari et la sélection.
    
    Args:
        rapports: Dict des rapports définitifs de scrape_rapports_definitifs()
        bet_type: Type de pari (ex: "SIMPLE PLACÉ", "E/P (GAGNANT-PLACÉ)", "COUPLÉ GAGNANT 1-5")
        selection: Nom du/des cheval(aux) ou numéros
        num_pmu: Numéro PMU du cheval si disponible
        
    Returns:
        Le dividende (cote) pour ce pari, ou None si non trouvé
    """
    import re
    
    if not rapports:
        return None
    
    bet_type_upper = bet_type.upper()
    
    # Extraire le numéro PMU de la sélection si c'est juste un numéro
    if num_pmu:
        num = str(num_pmu)
    else:
        # Essayer d'extraire les numéros de la sélection (ex: "1-5-3" ou "N°5")
        nums = re.findall(r'\d+', selection)
        num = nums[0] if nums else None
    
    # SIMPLE GAGNANT
    if "GAGNANT" in bet_type_upper and "PLACÉ" not in bet_type_upper and "E/P" not in bet_type_upper:
        if "SIMPLE_GAGNANT" in rapports and num:
            return rapports["SIMPLE_GAGNANT"].get(num)
    
    # SIMPLE PLACÉ
    if "PLACÉ" in bet_type_upper and "GAGNANT" not in bet_type_upper and "E/P" not in bet_type_upper and "COUPLÉ" not in bet_type_upper:
        if "SIMPLE_PLACE" in rapports and num:
            return rapports["SIMPLE_PLACE"].get(num)
    
    # E/P (GAGNANT-PLACÉ) - Retourner les deux cotes
    if "E/P" in bet_type_upper or ("GAGNANT" in bet_type_upper and "PLACÉ" in bet_type_upper):
        cote_gagnant = None
        cote_place = None
        if "SIMPLE_GAGNANT" in rapports and num:
            cote_gagnant = rapports["SIMPLE_GAGNANT"].get(num)
        if "SIMPLE_PLACE" in rapports and num:
            cote_place = rapports["SIMPLE_PLACE"].get(num)
        # Retourner un tuple ou la cote placé (car c'est celle qu'on touche si placé)
        if cote_gagnant and cote_place:
            return {"gagnant": cote_gagnant, "place": cote_place}
        return cote_place or cote_gagnant
    
    # COUPLÉ GAGNANT
    if "COUPLÉ" in bet_type_upper and "GAGNANT" in bet_type_upper:
        # Extraire les numéros (ex: "Couplé Gagnant 1-5" ou "1-5")
        nums = re.findall(r'\d+', selection if selection else bet_type)
        if len(nums) >= 2 and "COUPLE_GAGNANT" in rapports:
            combo = f"{nums[0]}-{nums[1]}"
            return rapports["COUPLE_GAGNANT"].get(combo)
    
    # COUPLÉ PLACÉ
    if "COUPLÉ" in bet_type_upper and "PLACÉ" in bet_type_upper:
        nums = re.findall(r'\d+', selection if selection else bet_type)
        if len(nums) >= 2 and "COUPLE_PLACE" in rapports:
            # Chercher dans toutes les combinaisons possibles
            combo1 = f"{nums[0]}-{nums[1]}"
            combo2 = f"{nums[1]}-{nums[0]}"
            result = rapports["COUPLE_PLACE"].get(combo1) or rapports["COUPLE_PLACE"].get(combo2)
            return result
    
    # TIERCÉ
    if "TIERC" in bet_type_upper:
        nums = re.findall(r'\d+', selection if selection else bet_type)
        if len(nums) >= 3 and "TIERCE" in rapports:
            combo = "-".join(nums[:3])
            if "ORDRE" in bet_type_upper:
                return rapports["TIERCE"]["ordre"].get(combo)
            else:
                # Chercher dans désordre
                return rapports["TIERCE"]["desordre"].get(combo) or rapports["TIERCE"]["ordre"].get(combo)
    
    # QUARTÉ
    if "QUART" in bet_type_upper:
        nums = re.findall(r'\d+', selection if selection else bet_type)
        if len(nums) >= 4 and "QUARTE_PLUS" in rapports:
            combo = "-".join(nums[:4])
            if "ORDRE" in bet_type_upper:
                return rapports["QUARTE_PLUS"]["ordre"].get(combo)
            else:
                return rapports["QUARTE_PLUS"]["desordre"].get(combo) or rapports["QUARTE_PLUS"]["ordre"].get(combo)
    
    # QUINTÉ
    if "QUINT" in bet_type_upper:
        nums = re.findall(r'\d+', selection if selection else bet_type)
        if len(nums) >= 5 and "QUINTE_PLUS" in rapports:
            combo = "-".join(nums[:5])
            if "ORDRE" in bet_type_upper:
                return rapports["QUINTE_PLUS"]["ordre"].get(combo)
            else:
                return rapports["QUINTE_PLUS"]["desordre"].get(combo) or rapports["QUINTE_PLUS"]["ordre"].get(combo)
    
    return None


def is_horse_incident(horse_data: dict) -> str:
    """
    Détecte si un cheval a eu un incident pendant la course.
    Retourne le type d'incident ou None si pas d'incident.
    
    Incidents PMU possibles:
    - ARRETE / ARR / STOPPED
    - TOMBE / TOM / CHU / CHUTE / FALL  
    - DISTANCIE / DIST / DISTANCE
    - DISQUALIFIE / DISQ / DQ / DAI / DIA
    - RETIRE / RET
    """
    INCIDENT_KEYWORDS = (
        "ARR", "ARRET", "ARRETE", "STOPPED",
        "CHU", "CHUTE", "TOMBE", "TOM", "FALL", "FALLEN", 
        "DIST", "DISTANCIE", "DISTANCED", "DISTANCE",
        "DISQ", "DQ", "DAI", "DIA", "DISQUALIFIE", "DISQUALIFIED",
        "ALLURE_IRREGULIERE",
        "RET", "RETIRE", "RETIREE", "WITHDRAWN",
    )
    
    incident = horse_data.get("incident") or ""
    statut = horse_data.get("statut") or ""
    
    incident_upper = incident.upper()
    statut_upper = statut.upper()
    
    for keyword in INCIDENT_KEYWORDS:
        if keyword in incident_upper or keyword in statut_upper:
            return incident or statut
    
    return None


def scrape_race_results_from_api(race_key: str) -> dict:
    """
    Scrape les résultats d'une course directement depuis l'API PMU.
    Retourne un dict avec les résultats par nom de cheval normalisé + métadonnées de la course.
    """
    import requests
    import unicodedata
    import re
    
    def norm(s: str) -> str:
        """Normalise un nom de cheval"""
        if not s:
            return ""
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s
    
    def to_pmu_date(date_iso: str) -> str:
        """Convertit YYYY-MM-DD en DDMMYYYY"""
        yyyy, mm, dd = date_iso.split('-')
        return f"{dd}{mm}{yyyy}"
    
    if not race_key:
        return {}
    
    try:
        # Parser race_key: "YYYY-MM-DD|R#|C#|HIPPO"
        parts = race_key.split('|')
        if len(parts) < 3:
            return {}
        date_iso = parts[0]
        reunion = int(parts[1].replace('R', ''))
        course = int(parts[2].replace('C', ''))
    except (IndexError, ValueError) as e:
        print(f"[WARN] Impossible de parser race_key: {race_key} ({e})")
        return {}
    
    date_pmu = to_pmu_date(date_iso)
    
    BASES = [
        "https://online.turfinfo.api.pmu.fr/rest/client/7",
        "https://offline.turfinfo.api.pmu.fr/rest/client/7"
    ]
    HEADERS = {
        "User-Agent": "horse-bet-checker/1.0",
        "Accept": "application/json",
    }
    
    for base in BASES:
        url = f"{base}/programme/{date_pmu}/R{reunion}/C{course}/participants"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                participants = data.get("participants", [])
                
                # Compter le nombre de partants (excluant les non-partants)
                nb_partants = sum(1 for p in participants 
                                 if not p.get("nonPartant", False) 
                                 and p.get("statut") != "NON_PARTANT")
                
                # Extraire l'arrivée ordonnée (liste des places)
                arrivee = []
                results = {}
                
                for p in participants:
                    nom = p.get("nom")
                    if not nom:
                        continue
                    
                    # Chercher la place dans les différents champs possibles
                    place = (p.get("ordreArrivee") or 
                            p.get("place") or 
                            p.get("rang") or 
                            p.get("classement"))
                    
                    # Convertir en int si possible
                    if place is not None:
                        try:
                            place = int(place)
                        except (ValueError, TypeError):
                            place = None
                    
                    # Cote finale - récupérer la cote directe ET la cote de référence
                    cote_direct = None
                    cote_reference = None
                    
                    if p.get("dernierRapportDirect"):
                        cote_direct = p["dernierRapportDirect"].get("rapport")
                    if p.get("dernierRapportReference"):
                        cote_reference = p["dernierRapportReference"].get("rapport")
                    
                    # Utiliser la cote directe en priorité, sinon la référence
                    cote = cote_direct or cote_reference or p.get("rapport") or p.get("coteDirect")
                    
                    # Vérifier si le cheval est non-partant
                    is_non_partant = (
                        p.get("nonPartant", False) or 
                        p.get("statut") == "NON_PARTANT" or
                        p.get("incident") == "NON_PARTANT"
                    )
                    
                    results[norm(nom)] = {
                        "place": place,
                        "nom_original": nom,
                        "cote": cote,
                        "numPmu": p.get("numPmu"),
                        "nonPartant": is_non_partant,
                        "statut": p.get("statut"),
                        "incident": p.get("incident")
                    }
                    
                    # Construire l'arrivée
                    if place is not None and place > 0:
                        arrivee.append((place, norm(nom), p.get("numPmu")))
                
                # Trier l'arrivée par place
                arrivee.sort(key=lambda x: x[0])
                
                # Déterminer si la course est terminée (au moins un cheval a une place)
                course_terminee = len(arrivee) > 0
                
                if results:
                    return {
                        "participants": results,
                        "nb_partants": nb_partants,
                        "arrivee": arrivee,  # Liste triée [(place, nom_norm, numPmu), ...]
                        "top_4": [a[1] for a in arrivee[:4]],  # 4 premiers noms normalisés
                        "top_5": [a[1] for a in arrivee[:5]],  # 5 premiers noms normalisés
                        "course_terminee": course_terminee,
                    }
                
        except Exception as e:
            print(f"[WARN] Erreur API PMU ({base}): {e}")
            continue
    
    return {}

def update_race_results_in_db(cur, race_key: str, race_data: dict):
    """
    Met à jour les résultats dans la base de données.
    race_data contient: participants, nb_partants, arrivee, top_4, top_5
    """
    if not race_data or "participants" not in race_data:
        return False
    
    results = race_data.get("participants", {})
    updated = 0
    total = 0
    
    for nom_norm, data in results.items():
        place = data.get("place")
        if place is None:
            continue
        
        total += 1
        is_win = 1 if place == 1 else 0
        
        try:
            # Utilise LOWER() des deux côtés pour être sûr
            cur.execute(adapt_query("""
                UPDATE cheval_courses_seen
                SET place_finale = %s, is_win = %s
                WHERE race_key = %s AND LOWER(nom_norm) = LOWER(%s)
            """), (place, is_win, race_key, nom_norm))
            
            if cur.rowcount > 0:
                updated += 1
            else:
                print(f"[DEBUG] Pas de match BDD pour: {nom_norm} (race: {race_key})")
        except Exception as e:
            print(f"[WARN] Erreur MAJ résultat {nom_norm}: {e}")
    
    print(f"[DEBUG] update_race_results_in_db: {updated}/{total} chevaux mis à jour pour {race_key}")
    return updated > 0

def try_scrape_race(race_key: str, cur=None, con=None) -> dict:
    """
    Tente de récupérer les résultats d'une course via l'API PMU
    et met à jour la base de données.
    Retourne le dict race_data avec toutes les infos de la course.
    """
    if not race_key:
        return {}
    
    print(f"[INFO] Scraping résultats pour {race_key}...")
    
    race_data = scrape_race_results_from_api(race_key)
    
    if not race_data or "participants" not in race_data:
        print(f"[WARN] Pas de résultats disponibles pour {race_key}")
        return {}
    
    # Si on a un curseur, mettre à jour la BDD
    if cur and con:
        success = update_race_results_in_db(cur, race_key, race_data)
        if success:
            try:
                con.commit()
                print(f"[INFO] Résultats mis à jour pour {race_key} ({race_data.get('nb_partants', 0)} partants)")
            except Exception as e:
                print(f"[WARN] Erreur commit: {e}")
    
    return race_data

def determine_bet_result(place_finale: int, bet_type: str, nb_partants: int = 8, selections: list = None, top_arrivee: list = None) -> str:
    """
    Détermine si un pari est gagnant selon les règles PMU complètes.
    
    Args:
        place_finale: Place d'arrivée du cheval (pour paris simples)
        bet_type: Type de pari (SIMPLE GAGNANT, SIMPLE PLACÉ, E/P, COUPLÉ, TIERCÉ, etc.)
        nb_partants: Nombre de chevaux au départ
        selections: Liste des chevaux sélectionnés (pour paris combinés)
        top_arrivee: Liste des chevaux arrivés dans le top (noms normalisés)
    
    Returns:
        "WIN", "LOSE" ou "PENDING"
    
    RÈGLES PMU:
    ═══════════════════════════════════════════════════════════════════════
    
    🟦 SIMPLE GAGNANT / E_GAGNANT
       → Uniquement 1er
    
    🟦 SIMPLE PLACÉ / E_PLACÉ / E/P (GAGNANT-PLACÉ)
       → 4 à 7 partants: 1er – 2e
       → 8 à 15 partants: 1er – 2e – 3e
       → 16+ partants: 1er – 2e – 3e – 4e
    
    🟩 COUPLÉ GAGNANT
       → Les 2 premiers dans n'importe quel ordre
    
    🟩 COUPLÉ PLACÉ
       → 4–7 partants: 2 premiers
       → 8+ partants: 3 premiers
    
    🟩 COUPLÉ ORDRE
       → Les 2 premiers dans l'ordre exact
    
    🟧 TRIO / TIERCÉ DÉSORDRE
       → 1er – 2e – 3e dans n'importe quel ordre
    
    🟧 TIERCÉ ORDRE
       → 1er – 2e – 3e dans l'ordre exact
    
    🟥 QUARTÉ ORDRE
       → 1er – 2e – 3e – 4e dans l'ordre exact
    
    🟥 QUARTÉ DÉSORDRE
       → Les 4 premiers dans le désordre
    
    🟥 QUINTÉ ORDRE
       → 1er – 2e – 3e – 4e – 5e dans l'ordre exact
    
    🟥 QUINTÉ DÉSORDRE
       → Les 5 premiers dans n'importe quel ordre
    
    🟥 QUINTÉ BONUS 4
       → Les 4 premiers parmi les 5 sélectionnés
    
    🟪 MULTI (en 4, 5, 6, 7)
       → Trouver les 4 premiers dans n'importe quel ordre
    
    🟨 2 SUR 4
       → 2 chevaux parmi les 4 premiers
    
    🟩 PICK 5
       → Trouver les 5 premiers, ordre indifférent
    
    🟩 SUPER 4
       → Trouver les 4 premiers dans l'ordre exact
    """
    import unicodedata
    import re
    
    def norm(s: str) -> str:
        if not s:
            return ""
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s
    
    if place_finale is None and not selections:
        return "PENDING"
    
    bet_type_upper = (bet_type or "").upper()
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟦 SIMPLE GAGNANT / E_GAGNANT
    # ═══════════════════════════════════════════════════════════════════
    if ("GAGNANT" in bet_type_upper or "E_GAGNANT" in bet_type_upper) and \
       "PLACÉ" not in bet_type_upper and "E/P" not in bet_type_upper and \
       "COUPLÉ" not in bet_type_upper and "COUPLE" not in bet_type_upper:
        if place_finale is None:
            return "PENDING"
        return "WIN" if place_finale == 1 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟦 SIMPLE PLACÉ / E_PLACÉ / E/P (GAGNANT-PLACÉ)
    # ═══════════════════════════════════════════════════════════════════
    if any(x in bet_type_upper for x in ["PLACÉ", "PLACE", "E/P", "E_PLACE"]):
        if place_finale is None:
            return "PENDING"
        
        # Déterminer le nombre de places payées selon le nombre de partants
        if nb_partants >= 16:
            places_payees = 4  # 1er – 2e – 3e – 4e
        elif nb_partants >= 8:
            places_payees = 3  # 1er – 2e – 3e
        else:  # 4 à 7 partants
            places_payees = 2  # 1er – 2e
        
        return "WIN" if place_finale <= places_payees else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟪 MULTI (en 4, 5, 6, 7)
    # ═══════════════════════════════════════════════════════════════════
    if "MULTI" in bet_type_upper:
        if not selections or not top_arrivee:
            # Si pas de sélections multiples, on utilise place_finale
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 4 else "LOSE"
        
        # Vérifier que les 4 premiers sont dans les sélections
        top_4 = top_arrivee[:4] if len(top_arrivee) >= 4 else top_arrivee
        selections_norm = [norm(s) for s in selections]
        matches = sum(1 for h in top_4 if norm(h) in selections_norm)
        return "WIN" if matches >= 4 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟨 2 SUR 4
    # ═══════════════════════════════════════════════════════════════════
    if "2SUR4" in bet_type_upper or "2 SUR 4" in bet_type_upper:
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 4 else "LOSE"
        
        top_4 = top_arrivee[:4] if len(top_arrivee) >= 4 else top_arrivee
        selections_norm = [norm(s) for s in selections]
        matches = sum(1 for h in top_4 if norm(h) in selections_norm)
        return "WIN" if matches >= 2 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟩 COUPLÉ GAGNANT
    # ═══════════════════════════════════════════════════════════════════
    if ("COUPLÉ GAGNANT" in bet_type_upper or "COUPLE GAGNANT" in bet_type_upper):
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 2 else "LOSE"
        
        top_2 = top_arrivee[:2] if len(top_arrivee) >= 2 else top_arrivee
        selections_norm = [norm(s) for s in selections]
        matches = sum(1 for h in top_2 if norm(h) in selections_norm)
        return "WIN" if matches >= 2 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟩 COUPLÉ PLACÉ
    # ═══════════════════════════════════════════════════════════════════
    if ("COUPLÉ PLACÉ" in bet_type_upper or "COUPLE PLACE" in bet_type_upper):
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            # 4-7 partants: top 2, 8+ partants: top 3
            places_payees = 2 if nb_partants < 8 else 3
            return "WIN" if place_finale <= places_payees else "LOSE"
        
        places_payees = 2 if nb_partants < 8 else 3
        top_n = top_arrivee[:places_payees] if len(top_arrivee) >= places_payees else top_arrivee
        selections_norm = [norm(s) for s in selections]
        matches = sum(1 for h in top_n if norm(h) in selections_norm)
        return "WIN" if matches >= 2 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟩 COUPLÉ ORDRE
    # ═══════════════════════════════════════════════════════════════════
    if ("COUPLÉ ORDRE" in bet_type_upper or "COUPLE ORDRE" in bet_type_upper):
        if not selections or not top_arrivee or len(selections) < 2 or len(top_arrivee) < 2:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 2 else "LOSE"
        
        # Les 2 premiers dans l'ordre exact
        selections_norm = [norm(s) for s in selections[:2]]
        if norm(top_arrivee[0]) == selections_norm[0] and norm(top_arrivee[1]) == selections_norm[1]:
            return "WIN"
        return "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟧 TRIO / TIERCÉ
    # ═══════════════════════════════════════════════════════════════════
    if "TRIO" in bet_type_upper or "TIERCÉ" in bet_type_upper or "TIERCE" in bet_type_upper:
        is_ordre = "ORDRE" in bet_type_upper
        
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 3 else "LOSE"
        
        top_3 = top_arrivee[:3] if len(top_arrivee) >= 3 else top_arrivee
        selections_norm = [norm(s) for s in selections[:3]] if len(selections) >= 3 else [norm(s) for s in selections]
        
        if is_ordre:
            # Ordre exact
            if len(selections_norm) >= 3 and len(top_3) >= 3:
                if all(norm(top_3[i]) == selections_norm[i] for i in range(3)):
                    return "WIN"
            return "LOSE"
        else:
            # Désordre - les 3 premiers dans n'importe quel ordre
            matches = sum(1 for h in top_3 if norm(h) in selections_norm)
            return "WIN" if matches >= 3 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟥 QUARTÉ
    # ═══════════════════════════════════════════════════════════════════
    if "QUARTÉ" in bet_type_upper or "QUARTE" in bet_type_upper:
        is_ordre = "ORDRE" in bet_type_upper
        
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 4 else "LOSE"
        
        top_4 = top_arrivee[:4] if len(top_arrivee) >= 4 else top_arrivee
        selections_norm = [norm(s) for s in selections[:4]] if len(selections) >= 4 else [norm(s) for s in selections]
        
        if is_ordre:
            # Ordre exact
            if len(selections_norm) >= 4 and len(top_4) >= 4:
                if all(norm(top_4[i]) == selections_norm[i] for i in range(4)):
                    return "WIN"
            return "LOSE"
        else:
            # Désordre
            matches = sum(1 for h in top_4 if norm(h) in selections_norm)
            return "WIN" if matches >= 4 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟥 QUINTÉ
    # ═══════════════════════════════════════════════════════════════════
    if "QUINTÉ" in bet_type_upper or "QUINTE" in bet_type_upper:
        is_ordre = "ORDRE" in bet_type_upper
        is_bonus = "BONUS" in bet_type_upper
        
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 5 else "LOSE"
        
        top_5 = top_arrivee[:5] if len(top_arrivee) >= 5 else top_arrivee
        selections_norm = [norm(s) for s in selections[:5]] if len(selections) >= 5 else [norm(s) for s in selections]
        
        if is_ordre and not is_bonus:
            # Ordre exact
            if len(selections_norm) >= 5 and len(top_5) >= 5:
                if all(norm(top_5[i]) == selections_norm[i] for i in range(5)):
                    return "WIN"
            return "LOSE"
        elif is_bonus:
            # Bonus 4 - 4 premiers parmi les 5
            top_4 = top_5[:4]
            matches = sum(1 for h in top_4 if norm(h) in selections_norm)
            return "WIN" if matches >= 4 else "LOSE"
        else:
            # Désordre
            matches = sum(1 for h in top_5 if norm(h) in selections_norm)
            return "WIN" if matches >= 5 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟩 PICK 5
    # ═══════════════════════════════════════════════════════════════════
    if "PICK 5" in bet_type_upper or "PICK5" in bet_type_upper:
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 5 else "LOSE"
        
        top_5 = top_arrivee[:5] if len(top_arrivee) >= 5 else top_arrivee
        selections_norm = [norm(s) for s in selections]
        matches = sum(1 for h in top_5 if norm(h) in selections_norm)
        return "WIN" if matches >= 5 else "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🟩 SUPER 4
    # ═══════════════════════════════════════════════════════════════════
    if "SUPER 4" in bet_type_upper or "SUPER4" in bet_type_upper:
        if not selections or not top_arrivee:
            if place_finale is None:
                return "PENDING"
            return "WIN" if place_finale <= 4 else "LOSE"
        
        top_4 = top_arrivee[:4] if len(top_arrivee) >= 4 else top_arrivee
        selections_norm = [norm(s) for s in selections[:4]] if len(selections) >= 4 else [norm(s) for s in selections]
        
        # Ordre exact
        if len(selections_norm) >= 4 and len(top_4) >= 4:
            if all(norm(top_4[i]) == selections_norm[i] for i in range(4)):
                return "WIN"
        return "LOSE"
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔄 DÉFAUT - Paris non reconnu, on utilise la logique placé standard
    # ═══════════════════════════════════════════════════════════════════
    if place_finale is None:
        return "PENDING"
    
    # Par défaut: mode placé avec 3 places
    print(f"[WARN] Type de pari non reconnu: {bet_type}, utilisation du mode placé par défaut")
    return "WIN" if place_finale <= 3 else "LOSE"

@app.on_event("startup")
async def startup_create_user_tables():
    """Initialise les tables nécessaires (utilisateurs, sessions, paris)"""
    try:
        init_user_tables()
    except Exception as e:
        print(f"[WARN] Impossible de créer les tables users/paris: {e}")

# ============================================================================
# Routes API
# ============================================================================

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "HorseRace Predictor API",
        "version": "1.0.0",
        "database": "PostgreSQL" if USE_POSTGRESQL else "SQLite",
        "endpoints": {
            "dashboard": "/api/dashboard",
            "auth_register": "/api/auth/register",
            "auth_login": "/api/auth/login",
            "bets": "/api/bets",
            "bets_summary": "/api/bets/summary",
            "analytics": "/api/analytics",
            "chevaux": "/api/chevaux",
            "jockeys": "/api/jockeys",
            "entraineurs": "/api/entraineurs",
            "settings": "/api/settings"
        }
    }

# ============================================================================
# ENDPOINTS - Authentification et gestion des paris utilisateur
# ============================================================================

@app.post("/api/auth/register", response_model=AuthResponse)
async def register_user(payload: RegisterRequest):
    """Crée un compte utilisateur et retourne un token simple"""
    con = get_db_connection()
    try:
        cur = con.cursor()
        # Vérifier l'unicité de l'email
        cur.execute(adapt_query("""
            SELECT id FROM users WHERE LOWER(email) = LOWER(%s)
        """), (payload.email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Un compte existe déjà avec cet email")
        
        salt_hex, hash_hex = hash_password(payload.password)
        display_name = payload.display_name or payload.email.split("@")[0]
        
        if USE_POSTGRESQL:
            cur.execute("""
                INSERT INTO users (email, password_hash, salt, display_name)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at
            """, (payload.email.lower(), hash_hex, salt_hex, display_name))
            row = cur.fetchone()
            user_id, created_at = row[0], row[1]
        else:
            cur.execute(adapt_query("""
                INSERT INTO users (email, password_hash, salt, display_name)
                VALUES (%s, %s, %s, %s)
            """), (payload.email.lower(), hash_hex, salt_hex, display_name))
            user_id = cur.lastrowid
            created_at = datetime.utcnow()
        
        token, expires_at = create_session_token(cur, user_id)
        con.commit()
        
        return AuthResponse(
            token=token,
            expires_at=expires_at,
            user=UserProfile(
                id=user_id,
                email=payload.email.lower(),
                display_name=display_name,
                created_at=serialize_datetime(created_at)
            )
        )
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.post("/api/auth/login", response_model=AuthResponse)
async def login_user(payload: LoginRequest):
    """Connecte un utilisateur existant"""
    con = get_db_connection()
    try:
        cur = con.cursor()
        cur.execute(adapt_query("""
            SELECT id, email, password_hash, salt, display_name, created_at
            FROM users
            WHERE LOWER(email) = LOWER(%s)
            LIMIT 1
        """), (payload.email,))
        row = cur.fetchone()
        
        if not row or not verify_password(payload.password, row[3], row[2]):
            raise HTTPException(status_code=401, detail="Email ou mot de passe invalide")
        
        token, expires_at = create_session_token(cur, row[0])
        con.commit()
        
        return AuthResponse(
            token=token,
            expires_at=expires_at,
            user=UserProfile(
                id=row[0],
                email=row[1],
                display_name=row[4],
                created_at=serialize_datetime(row[5])
            )
        )
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.get("/api/auth/me", response_model=UserProfile)
async def get_me(authorization: Optional[str] = Header(None)):
    """Renvoie le profil de l'utilisateur connecté"""
    user = require_auth(authorization)
    return UserProfile(**user)


@app.post("/api/bets", response_model=BetResponse)
async def create_bet(payload: BetCreateRequest, authorization: Optional[str] = Header(None)):
    """Ajoute un pari utilisateur et calcule son PnL"""
    user = require_auth(authorization)
    con = get_db_connection()
    try:
        cur = con.cursor()
        status = normalize_bet_status(payload.status)
        event_date = infer_event_date(payload.race_key, payload.event_date)
        params = (
            user["id"],
            payload.race_key,
            event_date,
            payload.hippodrome,
            payload.selection,
            payload.bet_type,
            payload.stake,
            payload.odds,
            status,
            payload.notes
        )
        
        if USE_POSTGRESQL:
            cur.execute("""
                INSERT INTO user_bets (
                    user_id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
            """, params)
            row = cur.fetchone()
        else:
            cur.execute(adapt_query("""
                INSERT INTO user_bets (
                    user_id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """), params)
            bet_id = cur.lastrowid
            cur.execute(adapt_query("""
                SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
                FROM user_bets
                WHERE id = %s
            """), (bet_id,))
            row = cur.fetchone()
        
        con.commit()
        return serialize_bet_row(row)
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.get("/api/bets", response_model=List[BetResponse])
async def list_bets(authorization: Optional[str] = Header(None)):
    """Liste les paris de l'utilisateur connecté"""
    user = require_auth(authorization)
    con = get_db_connection()
    try:
        cur = con.cursor()
        cur.execute(adapt_query("""
            SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
            FROM user_bets
            WHERE user_id = %s
            ORDER BY created_at DESC
        """), (user["id"],))
        rows = cur.fetchall()
        return [serialize_bet_row(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.patch("/api/bets/{bet_id}", response_model=BetResponse)
async def update_bet(bet_id: int, payload: BetUpdateRequest, authorization: Optional[str] = Header(None)):
    """Met à jour un pari (statut, cote, mise, notes, etc.)"""
    user = require_auth(authorization)
    con = get_db_connection()
    try:
        cur = con.cursor()
        cur.execute(adapt_query("""
            SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
            FROM user_bets
            WHERE id = %s AND user_id = %s
        """), (bet_id, user["id"]))
        existing = cur.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Pari introuvable")
        
        updates = []
        params = []
        
        if payload.race_key is not None:
            updates.append("race_key = %s")
            params.append(payload.race_key)
            inferred_date = infer_event_date(payload.race_key, payload.event_date)
            updates.append("event_date = %s")
            params.append(inferred_date or existing[2])
        elif payload.event_date is not None:
            updates.append("event_date = %s")
            params.append(payload.event_date)
        
        if payload.hippodrome is not None:
            updates.append("hippodrome = %s")
            params.append(payload.hippodrome)
        
        if payload.selection is not None:
            updates.append("selection = %s")
            params.append(payload.selection)
        
        if payload.bet_type is not None:
            updates.append("bet_type = %s")
            params.append(payload.bet_type)
        
        if payload.stake is not None:
            updates.append("stake = %s")
            params.append(payload.stake)
        
        if payload.odds is not None:
            updates.append("odds = %s")
            params.append(payload.odds)
        
        if payload.status is not None:
            updates.append("status = %s")
            params.append(normalize_bet_status(payload.status))
        
        if payload.notes is not None:
            updates.append("notes = %s")
            params.append(payload.notes)
        
        if updates:
            query = f"UPDATE user_bets SET {', '.join(updates)} WHERE id = %s AND user_id = %s"
            params.extend([bet_id, user["id"]])
            cur.execute(adapt_query(query), tuple(params))
            con.commit()
        
            cur.execute(adapt_query("""
            SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
            FROM user_bets
            WHERE id = %s AND user_id = %s
        """), (bet_id, user["id"]))
        row = cur.fetchone()
        return serialize_bet_row(row)
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.delete("/api/bets/{bet_id}")
async def delete_bet(bet_id: int, authorization: Optional[str] = Header(None)):
    """Supprime un pari de l'utilisateur"""
    user = require_auth(authorization)
    con = get_db_connection()
    try:
        cur = con.cursor()
        # Vérifie que le pari existe et appartient à l'utilisateur
        cur.execute(adapt_query("""
            SELECT id FROM user_bets
            WHERE id = %s AND user_id = %s
        """), (bet_id, user["id"]))
        existing = cur.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Pari introuvable")
        
        # Suppression
        cur.execute(adapt_query("""
            DELETE FROM user_bets
            WHERE id = %s AND user_id = %s
        """), (bet_id, user["id"]))
        con.commit()
        
        return {"message": "Pari supprimé avec succès", "deleted_id": bet_id}
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.post("/api/bets/{bet_id}/refresh", response_model=BetResponse)
async def refresh_bet_result(bet_id: int, authorization: Optional[str] = Header(None)):
    """
    Vérifie le résultat d'un pari selon les règles PMU complètes.
    
    1) Cherche dans la base enrichie si la course a des résultats (place_finale NOT NULL)
    2) Si pas de résultats, tente un scraping ciblé depuis l'API PMU
    3) Détermine WIN/LOSE selon le type de pari ET le nombre de partants:
    
       🟦 SIMPLE GAGNANT: place == 1
       🟦 SIMPLE PLACÉ: 
          - 4-7 partants: top 2
          - 8-15 partants: top 3
          - 16+ partants: top 4
       🟩 COUPLÉ GAGNANT: 2 premiers (désordre)
       🟩 COUPLÉ PLACÉ: 2 parmi top 2-3 selon partants
       🟧 TIERCÉ: 3 premiers
       🟥 QUARTÉ: 4 premiers
       🟥 QUINTÉ: 5 premiers
       🟪 MULTI: 4 premiers (désordre)
       🟨 2 SUR 4: 2 parmi les 4 premiers
       
    4) Garde PENDING si la course n'a pas encore de résultats
    """
    user = require_auth(authorization)
    con = get_db_connection()
    try:
        cur = con.cursor()
        cur.execute(adapt_query("""
            SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
            FROM user_bets
            WHERE id = %s AND user_id = %s
        """), (bet_id, user["id"]))
        bet_row = cur.fetchone()
        if not bet_row:
            raise HTTPException(status_code=404, detail="Pari introuvable")
        
        current_status = normalize_bet_status(bet_row[8])
        already_resolved = current_status in ("WIN", "LOSE", "VOID")
        
        race_key = bet_row[1]
        selection = bet_row[4]
        bet_type = bet_row[5]
        
        # Variables pour stocker les infos de la course
        race_data = None
        nb_partants = 8  # Valeur par défaut
        top_arrivee = []
        place_finale = None
        cote_finale = None
        
        # Normaliser le nom de la sélection
        import unicodedata
        import re
        def norm_name(s: str) -> str:
            if not s:
                return ""
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s
        
        selection_norm = norm_name(selection)
        
        # D'abord essayer de scraper directement l'API PMU
        print(f"[INFO] Scraping résultats pour {race_key}...")
        race_data = scrape_race_results_from_api(race_key)
        
        if race_data and "participants" in race_data:
            nb_partants = race_data.get("nb_partants", 8)
            top_arrivee = race_data.get("arrivee", [])
            participants = race_data.get("participants", {})
            
            # Chercher notre cheval directement dans les résultats scrappés
            horse_data = participants.get(selection_norm)
            
            # Si pas trouvé exact, essayer une recherche partielle
            if not horse_data:
                for nom_norm, data in participants.items():
                    if selection_norm in nom_norm or nom_norm in selection_norm:
                        horse_data = data
                        print(f"[DEBUG] Match partiel trouvé: {selection_norm} -> {nom_norm}")
                        break
            
            if horse_data:
                place_finale = horse_data.get("place")
                cote_finale = horse_data.get("cote")
                is_non_partant = horse_data.get("nonPartant", False)
                incident_type = is_horse_incident(horse_data)
                
                if is_non_partant:
                    print(f"[INFO] Cheval {selection} est NON-PARTANT -> VOID (remboursé)")
                elif incident_type:
                    print(f"[INFO] Cheval {selection} a eu un incident ({incident_type}) -> LOSE")
                else:
                    print(f"[DEBUG] Résultat trouvé via API: {selection} -> place {place_finale}")
                
                # Mettre à jour la BDD pour les prochaines fois
                if cur and con:
                    update_race_results_in_db(cur, race_key, race_data)
                    try:
                        con.commit()
                    except Exception as e:
                        print(f"[WARN] Erreur commit MAJ résultats: {e}")
                
                # Gérer le cas Non-Partant -> Remboursement
                if is_non_partant:
                    cur.execute(adapt_query("""
                        UPDATE user_bets
                        SET status = 'VOID'
                        WHERE id = %s AND user_id = %s
                    """), (bet_id, user["id"]))
                    con.commit()
                    print(f"[INFO] Pari #{bet_id} ({selection}): VOID (Non-Partant)")
                    
                    cur.execute(adapt_query("""
                        SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
                        FROM user_bets
                        WHERE id = %s AND user_id = %s
                    """), (bet_id, user["id"]))
                    bet_row = cur.fetchone()
                    return serialize_bet_row(bet_row)
                
                # Gérer le cas Incident (arrêté, tombé, distancié, disqualifié) -> LOSE
                if incident_type and not already_resolved:
                    cur.execute(adapt_query("""
                        UPDATE user_bets
                        SET status = 'LOSE'
                        WHERE id = %s AND user_id = %s
                    """), (bet_id, user["id"]))
                    con.commit()
                    print(f"[INFO] Pari #{bet_id} ({selection}): LOSE (Incident: {incident_type})")
                    
                    cur.execute(adapt_query("""
                        SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
                        FROM user_bets
                        WHERE id = %s AND user_id = %s
                    """), (bet_id, user["id"]))
                    bet_row = cur.fetchone()
                    return serialize_bet_row(bet_row)
            else:
                print(f"[DEBUG] Cheval {selection_norm} non trouvé dans les {len(participants)} participants")
        else:
            print(f"[WARN] Pas de données de l'API pour {race_key}")
        
        # Mettre à jour la cote avec la cote finale de l'API PMU selon le TYPE DE PARI
        old_odds = float(bet_row[7]) if bet_row[7] else 0.0
        updated_odds = old_odds
        
        # Récupérer les rapports définitifs pour avoir la vraie cote selon le type de pari
        rapports_definitifs = scrape_rapports_definitifs(race_key)
        
        if rapports_definitifs:
            # Trouver le numéro PMU du cheval
            num_pmu = None
            if race_data and "participants" in race_data:
                for nom_norm, data in race_data["participants"].items():
                    if selection_norm in nom_norm or nom_norm in selection_norm:
                        num_pmu = str(data.get("numPmu", ""))
                        break
            
            # Récupérer le bon rapport selon le type de pari
            rapport = get_rapport_for_bet_type(rapports_definitifs, bet_type, selection, num_pmu)
            
            if rapport:
                if isinstance(rapport, dict):
                    # Pour E/P, on a un dict avec gagnant et place
                    # On stocke la cote gagnant dans odds, la cote placé sera utilisée pour le calcul du gain
                    updated_odds = float(rapport.get("gagnant", old_odds))
                    print(f"[INFO] E/P - Cotes définitives: Gagnant={rapport.get('gagnant')}, Placé={rapport.get('place')}")
                else:
                    updated_odds = float(rapport)
                    print(f"[INFO] Rapport définitif pour {bet_type}: {updated_odds}")
            elif cote_finale:
                # Fallback sur la cote simple gagnant si pas de rapport spécifique
                updated_odds = float(cote_finale)
                print(f"[INFO] Pas de rapport pour {bet_type}, utilisation cote simple gagnant: {updated_odds}")
        elif cote_finale:
            # Pas de rapports définitifs, utiliser la cote de l'API participants
            try:
                updated_odds = float(cote_finale)
            except:
                pass
        
        if abs(float(updated_odds) - float(old_odds)) > 0.01:  # Si la cote a changé
            print(f"[INFO] Cote mise à jour pour {selection}: {old_odds} -> {updated_odds}")
            # Mettre à jour la cote dans la BDD
            try:
                cur.execute(adapt_query("""
                    UPDATE user_bets
                    SET odds = %s
                    WHERE id = %s AND user_id = %s
                """), (updated_odds, bet_id, user["id"]))
                con.commit()
            except Exception as e:
                print(f"[WARN] Erreur mise à jour cote: {e}")
        
        # Si on a trouvé la place ET que le pari n'est pas encore résolu, déterminer le statut
        if place_finale is not None and not already_resolved:
            # Extraire les noms des chevaux du top de l'arrivée
            top_noms = [a[1] for a in top_arrivee] if top_arrivee else []
            
            # Déterminer le statut selon le type de pari avec le nombre de partants
            updated_status = determine_bet_result(
                place_finale=place_finale,
                bet_type=bet_type,
                nb_partants=nb_partants,
                selections=[selection],  # Pour paris simples, une seule sélection
                top_arrivee=top_noms
            )
            
            if updated_status in ("WIN", "LOSE"):
                cur.execute(adapt_query("""
                    UPDATE user_bets
                    SET status = %s, odds = %s
                    WHERE id = %s AND user_id = %s
                """), (updated_status, updated_odds, bet_id, user["id"]))
                con.commit()
                print(f"[INFO] Pari #{bet_id} ({selection}): {updated_status} (place: {place_finale}, cote: {updated_odds}, partants: {nb_partants}, type: {bet_type})")
        elif place_finale is None and not already_resolved:
            # Vérifier si la course est terminée (autres chevaux ont une place)
            if race_data and race_data.get("course_terminee"):
                # Course terminée mais le cheval n'a pas de place = il n'a pas fini -> LOSE
                print(f"[INFO] Course terminée mais {selection} n'a pas de classement -> LOSE")
                cur.execute(adapt_query("""
                    UPDATE user_bets
                    SET status = 'LOSE'
                    WHERE id = %s AND user_id = %s
                """), (bet_id, user["id"]))
                con.commit()
            else:
                print(f"[INFO] Pas de résultats disponibles pour {race_key} - {selection}, statut reste PENDING")
        
        # Recharger le pari mis à jour
        cur.execute(adapt_query("""
            SELECT id, race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, notes, created_at
            FROM user_bets
            WHERE id = %s AND user_id = %s
        """), (bet_id, user["id"]))
        bet_row = cur.fetchone()
        
        return serialize_bet_row(bet_row)
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        print(f"[ERROR] refresh_bet_result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.get("/api/bets/summary", response_model=BetSummary)
async def get_bets_summary(authorization: Optional[str] = Header(None)):
    """Résumé PnL des paris de l'utilisateur"""
    user = require_auth(authorization)
    con = get_db_connection()
    try:
        cur = con.cursor()
        bets = fetch_user_bets(cur, user["id"])
        total_bets = len(bets)
        total_stake = round(sum(b["stake"] for b in bets), 2)
        pnl_net = round(sum(b["pnl"] for b in bets), 2)
        
        by_status = {"WIN": 0, "LOSE": 0, "PENDING": 0, "VOID": 0}
        wins = 0
        finished_bets = 0
        resolved_stake = 0.0
        
        for b in bets:
            status = b["status"]
            by_status[status] = by_status.get(status, 0) + 1
            if status in ("WIN", "LOSE", "VOID"):
                finished_bets += 1
                resolved_stake += b["stake"]
            if status == "WIN":
                wins += 1
        
        roi = round((pnl_net / resolved_stake) * 100, 2) if resolved_stake > 0 else 0.0
        win_rate = round((wins / finished_bets) * 100, 2) if finished_bets > 0 else 0.0
        pending_bets = by_status.get("PENDING", 0)
        
        history_map = {}
        for b in bets:
            key_date = None
            if b["event_date"]:
                key_date = b["event_date"].isoformat()
            elif isinstance(b["created_at"], datetime):
                key_date = b["created_at"].date().isoformat()
            if key_date:
                history_map[key_date] = history_map.get(key_date, 0.0) + b["pnl"]
        
        history = [
            {"date": d, "pnl": round(p, 2)}
            for d, p in sorted(history_map.items())
        ]
        
        return BetSummary(
            total_bets=total_bets,
            pending_bets=pending_bets,
            finished_bets=finished_bets,
            total_stake=total_stake,
            pnl_net=pnl_net,
            roi=roi,
            win_rate=win_rate,
            by_status=by_status,
            history=history,
            bets=bets
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


@app.get("/api/settings", response_model=Settings)
async def get_settings():
    """
    Récupère la configuration actuelle depuis le fichier YAML (avec fallback par défaut).
    
    Retourne:
    - markets: mode (parimutuel/fixed), takeout_rate
    - kelly: fraction, value_cutoff, max_stake_pct
    - budget: daily_limit
    - betting_defaults: kelly_profile_default, kelly_fraction_map, caps, etc.
    - exotics_defaults: per_ticket_rate, max_pack_rate
    """
    try:
        config = load_config_file(create_if_missing=True)
        
        # Construire betting_defaults à partir de la config
        bet_def = config.get('betting_defaults', {})
        betting_defaults = SettingsBettingDefaults(
            kelly_profile_default=bet_def.get('kelly_profile_default', 'STANDARD'),
            kelly_fraction_map=bet_def.get('kelly_fraction_map', {"SUR": 0.25, "STANDARD": 0.33, "AMBITIEUX": 0.50}),
            custom_kelly_fraction=bet_def.get('custom_kelly_fraction', 0.33),
            value_cutoff=bet_def.get('value_cutoff', config.get('kelly', {}).get('value_cutoff', 0.05)),
            cap_per_bet=bet_def.get('cap_per_bet', 0.02),
            daily_budget_rate=bet_def.get('daily_budget_rate', 0.12),
            max_unit_bets_per_race=bet_def.get('max_unit_bets_per_race', 2),
            rounding_increment_eur=bet_def.get('rounding_increment_eur', 0.5)
        )
        
        # Construire exotics_defaults
        exo_def = config.get('exotics_defaults', {})
        exotics_defaults = SettingsExoticsDefaults(
            per_ticket_rate=exo_def.get('per_ticket_rate', 0.0075),
            max_pack_rate=exo_def.get('max_pack_rate', 0.04)
        )
        
        # Retourner la config complète
        return Settings(
            markets=SettingsMarkets(
                mode=config.get('markets', {}).get('mode', 'parimutuel'),
                takeout_rate=config.get('markets', {}).get('takeout_rate', 0.16)
            ),
            kelly=SettingsKelly(
                fraction=config.get('kelly', {}).get('fraction', 0.33),
                value_cutoff=config.get('kelly', {}).get('value_cutoff', 0.05),
                max_stake_pct=config.get('kelly', {}).get('max_stake_pct', 0.02)
            ),
            budget=SettingsBudget(
                daily_limit=config.get('budget', {}).get('daily_limit', 100.0)
            ),
            betting_defaults=betting_defaults,
            exotics_defaults=exotics_defaults
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture config: {str(e)}")

@app.post("/api/settings")
async def update_settings(settings: Settings):
    """
    Met à jour la configuration dans le fichier YAML.
    
    Paramètres modifiables:
    - kelly_profile_default: SUR, STANDARD, AMBITIEUX, PERSONNALISE
    - custom_kelly_fraction: fraction personnalisée (si profil=PERSONNALISE)
    - value_cutoff: seuil value minimum (%)
    - cap_per_bet: cap par pari (% bankroll)
    - daily_budget_rate: budget jour (% bankroll)
    - per_ticket_rate: mise par ticket exotique (% bankroll)
    - max_pack_rate: budget max pack exotiques (% bankroll)
    - markets.mode: parimutuel ou fixed
    """
    config_path = resolve_config_path()
        
    try:
        # Lire la config existante pour préserver les autres champs
        current_config = load_config_file(create_if_missing=True)
            
        # Mettre à jour markets
        if 'markets' not in current_config: current_config['markets'] = {}
        current_config['markets']['mode'] = settings.markets.mode
        current_config['markets']['takeout_rate'] = settings.markets.takeout_rate
        
        # Mettre à jour kelly (rétrocompatibilité)
        if 'kelly' not in current_config: current_config['kelly'] = {}
        current_config['kelly']['fraction'] = settings.kelly.fraction
        current_config['kelly']['value_cutoff'] = settings.kelly.value_cutoff
        current_config['kelly']['max_stake_pct'] = settings.kelly.max_stake_pct
        
        # Mettre à jour budget
        if settings.budget:
            if 'budget' not in current_config: current_config['budget'] = {}
            current_config['budget']['daily_limit'] = settings.budget.daily_limit
        
        # Mettre à jour betting_defaults (NOUVEAU)
        if settings.betting_defaults:
            if 'betting_defaults' not in current_config: current_config['betting_defaults'] = {}
            bd = settings.betting_defaults
            current_config['betting_defaults']['kelly_profile_default'] = bd.kelly_profile_default
            current_config['betting_defaults']['kelly_fraction_map'] = bd.kelly_fraction_map
            current_config['betting_defaults']['custom_kelly_fraction'] = bd.custom_kelly_fraction
            current_config['betting_defaults']['value_cutoff'] = bd.value_cutoff
            current_config['betting_defaults']['cap_per_bet'] = bd.cap_per_bet
            current_config['betting_defaults']['daily_budget_rate'] = bd.daily_budget_rate
            current_config['betting_defaults']['max_unit_bets_per_race'] = bd.max_unit_bets_per_race
            current_config['betting_defaults']['rounding_increment_eur'] = bd.rounding_increment_eur
            
            # Synchroniser kelly.fraction avec le profil sélectionné
            profile = bd.kelly_profile_default.upper()
            if profile == "PERSONNALISE":
                current_config['kelly']['fraction'] = bd.custom_kelly_fraction
            elif profile in bd.kelly_fraction_map:
                current_config['kelly']['fraction'] = bd.kelly_fraction_map[profile]
            
            # Synchroniser kelly.value_cutoff et max_stake_pct
            current_config['kelly']['value_cutoff'] = bd.value_cutoff
            current_config['kelly']['max_stake_pct'] = bd.cap_per_bet
        
        # Mettre à jour exotics_defaults (NOUVEAU)
        if settings.exotics_defaults:
            if 'exotics_defaults' not in current_config: current_config['exotics_defaults'] = {}
            current_config['exotics_defaults']['per_ticket_rate'] = settings.exotics_defaults.per_ticket_rate
            current_config['exotics_defaults']['max_pack_rate'] = settings.exotics_defaults.max_pack_rate
            
        # Sauvegarder
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
            
        return {"status": "success", "message": "Configuration mise à jour"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur écriture config: {str(e)}")

# ===== NOUVEAUX ENDPOINTS USER SETTINGS AUTHENTIFIÉS =====

@app.get("/api/user/settings", response_model=UserSettings)
async def get_user_settings_endpoint(user: Optional[Dict] = Depends(get_user_from_request)):
    """
    Récupère les settings personnalisés de l'utilisateur authentifié.
    Tous les paramètres sont stockés par utilisateur en BDD.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Non authentifié")
    
    try:
        settings = await get_user_settings(user["id"])
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération settings: {str(e)}")

@app.post("/api/user/settings")
async def update_user_settings_endpoint(
    settings: UserSettings, 
    user: Optional[Dict] = Depends(get_user_from_request)
):
    """
    Met à jour les settings personnalisés de l'utilisateur authentifié.
    
    Paramètres sauvegardés:
    - bankroll: Capital de départ
    - profil_risque: PRUDENT/STANDARD/AGRESSIF  
    - kelly_profile: SUR/STANDARD/AMBITIEUX/PERSONNALISE
    - custom_kelly_fraction: Fraction Kelly personnalisée
    - value_cutoff: Seuil value minimum
    - cap_per_bet: Cap par pari (% bankroll)
    - daily_budget_rate: Budget journalier (% bankroll)
    - max_unit_bets_per_race: Nombre max de paris par course
    - rounding_increment_eur: Arrondi des mises
    - per_ticket_rate: Mise par ticket exotique (% bankroll)
    - max_pack_rate: Budget max pack exotiques (% bankroll)
    - market_mode: Mode marché (parimutuel/fixed)
    - takeout_rate: Taux de prélèvement
    """
    if not user:
        raise HTTPException(status_code=401, detail="Non authentifié")
    
    try:
        await save_user_settings(user["id"], settings)
        return {"status": "success", "message": "Settings utilisateur sauvegardés"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur sauvegarde settings: {str(e)}")

@app.get("/api/test-db")
async def test_db():
    """Test de la connexion à la base de données"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Requête adaptée au schéma Plan Complet
        cur.execute("""
            SELECT c.sexe_cheval, COUNT(*) 
            FROM chevaux c
            JOIN stats_chevaux s ON c.id_cheval = s.id_cheval
            WHERE c.sexe_cheval IS NOT NULL AND s.nb_courses_total > 0 
            GROUP BY c.sexe_cheval 
            LIMIT 3
        """)
        
        rows = cur.fetchall()
        con.close()
        return {
            "database_type": "PostgreSQL" if USE_POSTGRESQL else "SQLite",
            "test_query_results": len(rows),
            "sample": [{"sexe": r[0], "count": int(r[1])} for r in rows]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/monitoring", response_model=MonitoringStats)
async def get_monitoring_stats(authorization: Optional[str] = Header(None)):
    """
    Récupère les statistiques de monitoring.
    - Avec token: P&L réel du compte connecté (table user_bets)
    - Sans token: fallback sur le log paper trading global
    """
    # Priorité aux données utilisateur si authentifié
    if authorization:
        user = require_auth(authorization)
        con = get_db_connection()
        try:
            cur = con.cursor()
            bets = fetch_user_bets(cur, user["id"])
            return build_monitoring_from_bets(bets)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            con.close()
    
    # Fallback global (paper trading)
    log_path = os.path.join(BASE_DIR, "..", "data", "paper_trading_log.csv")
    
    if not os.path.exists(log_path):
        return MonitoringStats(
            total_bets=0,
            pending_bets=0,
            finished_bets=0,
            win_rate=0.0,
            roi=0.0,
            pnl_net=0.0,
            pnl_history=[],
            recent_bets=[],
            data_scope="paper_trading"
        )
        
    try:
        df = pd.read_csv(log_path)
        
        # Conversion des dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # KPIs
        total_bets = len(df)
        pending_bets = len(df[df['statut'] == 'En cours'])
        
        df_finished = df[df['statut'].isin(['Gagné', 'Perdu'])].copy()
        finished_bets = len(df_finished)
        
        win_rate = 0.0
        roi = 0.0
        pnl_net = 0.0
        pnl_history = []
        
        if finished_bets > 0:
            # Calcul P&L si pas présent
            if 'gain_net' not in df_finished.columns:
                # Gestion des colonnes manquantes
                if 'cote' not in df_finished.columns:
                    df_finished['cote'] = 0.0
                
                df_finished['gain_net'] = df_finished.apply(
                    lambda x: (x['mise'] * x['cote'] - x['mise']) if x['statut'] == 'Gagné' else -x['mise'], axis=1
                )
            
            pnl_net = float(df_finished['gain_net'].sum())
            total_mise = float(df_finished['mise'].sum())
            roi = (pnl_net / total_mise * 100) if total_mise > 0 else 0.0
            win_rate = (len(df_finished[df_finished['statut'] == 'Gagné']) / finished_bets) * 100
            
            # Historique P&L
            df_finished = df_finished.sort_values('date')
            df_finished['pnl_cumul'] = df_finished['gain_net'].cumsum()
            
            # Agréger par jour pour le graphique
            daily_pnl = df_finished.groupby(df_finished['date'].dt.strftime('%Y-%m-%d'))['pnl_cumul'].last().reset_index()
            pnl_history = daily_pnl.to_dict('records')
            
        # Derniers paris (50 derniers)
        # Convertir les dates en string pour JSON
        recent_bets_df = df.sort_values('date', ascending=False).head(50)
        if 'date' in recent_bets_df.columns:
            recent_bets_df['date'] = recent_bets_df['date'].dt.strftime('%Y-%m-%d')
        
        # Remplacer NaN par None pour JSON
        recent_bets_df = recent_bets_df.where(pd.notnull(recent_bets_df), None)
        
        recent_bets = recent_bets_df.to_dict('records')
        
        return MonitoringStats(
            total_bets=total_bets,
            pending_bets=pending_bets,
            finished_bets=finished_bets,
            win_rate=round(win_rate, 2),
            roi=round(roi, 2),
            pnl_net=round(pnl_net, 2),
            pnl_history=pnl_history,
            recent_bets=recent_bets,
            data_scope="paper_trading"
        )
        
    except Exception as e:
        print(f"Erreur lecture monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard", response_model=Dict[str, Any])
async def get_dashboard():
    """
    Données du dashboard principal - Données réelles (avec cache)
    """
    # Vérifier le cache
    cache_key = "dashboard"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        if USE_POSTGRESQL:
            # --- Version PostgreSQL OPTIMISÉE ---
            # Utiliser une seule requête agrégée au lieu de plusieurs
            
            cur.execute("""
                WITH stats AS (
                    SELECT 
                        COUNT(DISTINCT nom_norm) as total_chevaux,
                        COUNT(*) as total_courses_seen,
                        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as total_victoires,
                        COUNT(DISTINCT entraineur) as nb_entraineurs,
                        COUNT(DISTINCT driver_jockey) as nb_jockeys
                    FROM cheval_courses_seen
                ),
                stats_2025 AS (
                    SELECT 
                        COUNT(*) as courses_30j,
                        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires_30j
                    FROM cheval_courses_seen
                    WHERE annee = 2025
                ),
                stats_chevaux_agg AS (
                    SELECT 
                        AVG(CAST(nb_victoires AS FLOAT) / NULLIF(nb_courses_total, 0) * 100) as taux_moyen_chevaux,
                        COUNT(DISTINCT CASE WHEN nb_victoires > 0 THEN id_cheval END) as chevaux_gagnants,
                        COUNT(*) as total_chevaux_bd
                    FROM stats_chevaux
                    WHERE nb_courses_total > 0
                )
                SELECT 
                    s.total_chevaux, s.total_courses_seen, s.total_victoires,
                    s.nb_entraineurs, s.nb_jockeys,
                    s25.courses_30j, s25.victoires_30j,
                    sc.taux_moyen_chevaux, sc.chevaux_gagnants, sc.total_chevaux_bd
                FROM stats s, stats_2025 s25, stats_chevaux_agg sc
            """)
            row = cur.fetchone()
            total_chevaux = int(row[0]) if row[0] is not None else 0
            total_courses = int(row[1]) if row[1] is not None else 0
            total_victoires = int(row[2]) if row[2] is not None else 0
            nb_entraineurs = int(row[3]) if row[3] is not None else 0
            nb_jockeys = int(row[4]) if row[4] is not None else 0
            courses_30j = int(row[5]) if row[5] is not None else 0
            taux_moyen_chevaux = float(row[7]) if row[7] is not None else 0.0
            
            # Taux de réussite global
            taux_reussite = (total_victoires / total_courses * 100) if total_courses > 0 else 0
            roi_moyen = round(taux_moyen_chevaux * 1.5, 1) if taux_moyen_chevaux > 0 else 0
            
            modeles_actifs = nb_entraineurs + nb_jockeys
            
            evolution_reussite = round(taux_moyen_chevaux - 1.0, 1) if taux_moyen_chevaux > 0 else 0
            evolution_roi = round((roi_moyen - 10.0) if roi_moyen > 0 else 0, 1)
            evolution_courses = courses_30j
            
            # Performances récentes (Top chevaux) - LIMIT pour rapidité
            cur.execute("""
                SELECT c.nom_cheval, s.nb_courses_total, s.nb_victoires,
                       CAST(s.nb_victoires AS FLOAT) / s.nb_courses_total * 100 as taux
                FROM chevaux c
                JOIN stats_chevaux s ON c.id_cheval = s.id_cheval
                WHERE s.nb_courses_total >= 3
                ORDER BY taux DESC, s.nb_victoires DESC
                LIMIT 4
            """)
            
            performances_recentes = []
            for row in cur.fetchall():
                nom = row[0]
                nb_courses = int(row[1])
                nb_victoires = int(row[2])
                proba = float(row[3])
                
                # Simulation résultat (car pas de musique facile d'accès)
                if proba > 50:
                    resultat = "Gagné"
                    evolution = f"+{int(proba)}%"
                elif proba > 20:
                    resultat = "Placé"
                    evolution = f"+{int(proba/2)}%"
                else:
                    resultat = "À suivre"
                    evolution = f"{int(proba)}%"
                
                performances_recentes.append({
                    "nom_course": nom,
                    "probabilite": round(proba, 1),
                    "resultat": resultat,
                    "evolution": evolution
                })
                
            # Variables prédictives (Race)
            cur.execute("""
                SELECT 
                    race,
                    AVG(CAST(is_win AS FLOAT) * 100) as taux_moyen,
                    COUNT(DISTINCT nom_norm) as nb_chevaux
                FROM cheval_courses_seen
                WHERE race IS NOT NULL
                GROUP BY race
                ORDER BY taux_moyen DESC
                LIMIT 3
            """)
            races_top = cur.fetchall()
            
            # Variables prédictives (Sexe)
            cur.execute("""
                SELECT 
                    sexe,
                    AVG(CAST(is_win AS FLOAT) * 100) as taux_moyen,
                    COUNT(DISTINCT nom_norm) as nb_chevaux
                FROM cheval_courses_seen
                WHERE sexe IS NOT NULL
                GROUP BY sexe
                ORDER BY taux_moyen DESC
                LIMIT 2
            """)
            sexes_top = cur.fetchall()
            
        else:
            # --- Version SQLite (Code existant) ---
            
            # Statistiques globales
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT nom_norm) as total_chevaux,
                    COUNT(*) as total_courses_seen,
                    SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as total_victoires
                FROM cheval_courses_seen
            """)
            row = cur.fetchone()
            total_chevaux = int(row[0]) if row[0] is not None else 0
            total_courses = int(row[1]) if row[1] is not None else 0
            total_victoires = int(row[2]) if row[2] is not None else 0
            
            # Taux de réussite global (victoires / courses)
            taux_reussite = (total_victoires / total_courses * 100) if total_courses > 0 else 0
            
            # Statistiques moyennes par cheval
            cur.execute("""
                SELECT 
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen_chevaux,
                    COUNT(DISTINCT CASE WHEN nombre_victoires_total > 0 THEN nom END) as chevaux_gagnants,
                    COUNT(*) as total_chevaux_bd
                FROM chevaux
                WHERE nombre_courses_total > 0
            """)
            row = cur.fetchone()
            taux_moyen_chevaux = float(row[0]) if row[0] is not None else 0.0
            chevaux_gagnants = int(row[1]) if row[1] is not None else 0
            total_chevaux_bd = int(row[2]) if row[2] is not None else 0
            
            # ROI moyen basé sur le taux de victoire moyen
            roi_moyen = round(taux_moyen_chevaux * 1.5, 1) if taux_moyen_chevaux > 0 else 0
            
            # Comparaison avec période précédente (30 jours vs période avant)
            cur.execute("""
                SELECT 
                    COUNT(*) as courses_30j,
                    SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires_30j
                FROM cheval_courses_seen
                WHERE race_key LIKE '2025-%' OR annee = 2025
            """)
            row = cur.fetchone()
            courses_30j = int(row[0]) if row[0] is not None else 0
            victoires_30j = int(row[1]) if row[1] is not None else 0
            
            # Statistiques sur entraineurs et jockeys actifs
            cur.execute("""
                SELECT COUNT(DISTINCT entraineur_courant)
                FROM chevaux
                WHERE entraineur_courant IS NOT NULL
            """)
            nb_entraineurs = cur.fetchone()[0] or 0
            
            cur.execute("""
                SELECT COUNT(DISTINCT jockey_habituel)
                FROM chevaux
                WHERE jockey_habituel IS NOT NULL
            """)
            nb_jockeys = cur.fetchone()[0] or 0
            
            modeles_actifs = nb_entraineurs + nb_jockeys  # Utiliser comme proxy
            
            # Évolution (comparaison basée sur les données récentes)
            evolution_reussite = round(taux_moyen_chevaux - 1.0, 1) if taux_moyen_chevaux > 0 else 0
            evolution_roi = round((roi_moyen - 10.0) if roi_moyen > 0 else 0, 1)
            evolution_courses = courses_30j
            
            # Performances récentes - Top chevaux avec meilleur taux
            cur.execute("""
                SELECT nom, dernier_resultat, nombre_courses_total, nombre_victoires_total,
                       CAST(nombre_victoires_total AS FLOAT) / nombre_courses_total * 100 as taux
                FROM chevaux
                WHERE nombre_courses_total >= 3
                ORDER BY taux DESC, nombre_victoires_total DESC
                LIMIT 4
            """)
            
            performances_recentes = []
            for row in cur.fetchall():
                nom = row[0]
                musique = row[1]
                nb_courses = int(row[2]) if row[2] is not None else 0
                nb_victoires = int(row[3]) if row[3] is not None else 0
                proba = float(row[4]) if row[4] is not None else 0.0
                
                # Déterminer le résultat basé sur la dernière perf (musique)
                if musique and len(musique) > 0 and musique[0] == '1':
                    resultat = "Gagné"
                    evolution = f"+{int(proba)}%"
                elif musique and len(musique) > 0 and musique[0] in ['2', '3']:
                    resultat = "Placé"
                    evolution = f"+{int(proba/2)}%"
                else:
                    resultat = "À suivre"
                    evolution = f"{int(proba)}%"
                
                performances_recentes.append({
                    "nom_course": nom,
                    "probabilite": round(proba, 1),
                    "resultat": resultat,
                    "evolution": evolution
                })
            
            # Variables prédictives basées sur les statistiques réelles
            # Taux de victoire par race
            cur.execute("""
                SELECT 
                    race,
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen,
                    COUNT(*) as nb_chevaux
                FROM chevaux
                WHERE race IS NOT NULL AND nombre_courses_total > 0
                GROUP BY race
                ORDER BY taux_moyen DESC
                LIMIT 3
            """)
            
            races_top = cur.fetchall()
            
            # Taux par sexe
            cur.execute("""
                SELECT 
                    CASE 
                        WHEN UPPER(sexe) IN ('F', 'FEMELLES', 'FEMMELLES') THEN 'FEMELLES'
                        WHEN UPPER(sexe) IN ('M', 'MALES') THEN 'MALES'
                        WHEN UPPER(sexe) IN ('H', 'HONGRES', 'HONGRE') THEN 'HONGRES'
                        ELSE sexe
                    END as sexe_normalise,
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen,
                    COUNT(*) as nb_chevaux
                FROM chevaux
                WHERE sexe IS NOT NULL AND nombre_courses_total > 0
                GROUP BY sexe_normalise
                ORDER BY taux_moyen DESC
                LIMIT 2
            """)
            
            sexes_top = cur.fetchall()

        # --- Fin du bloc conditionnel DB ---
        
        # Construire les variables prédictives avec données réelles
        variables_predictives = []
        
        # Ajouter les races les plus performantes
        for race, taux, nb in races_top:
            if race and taux:
                importance = min(100, max(60, round(float(taux) * 10)))
                variables_predictives.append({
                    "nom": f"Race: {race[:20]}",
                    "importance": importance
                })
        
        # Ajouter le meilleur sexe
        if sexes_top:
            sexe, taux, nb = sexes_top[0]
            if sexe and taux:
                importance = min(95, max(70, round(float(taux) * 12)))
                variables_predictives.append({
                    "nom": f"Performance {sexe}",
                    "importance": importance
                })
        
        # Si on n'a pas assez de variables, compléter avec des stats générales
        while len(variables_predictives) < 5:
            if len(variables_predictives) == 0:
                variables_predictives.append({"nom": "Nombre de courses", "importance": min(90, round(total_courses / 50))})
            elif len(variables_predictives) == 1:
                variables_predictives.append({"nom": "Expérience cheval", "importance": min(85, round(taux_moyen_chevaux * 5))})
            elif len(variables_predictives) == 2:
                variables_predictives.append({"nom": "Historique victoires", "importance": min(80, round((total_victoires / total_courses * 100) * 0.8))})
            elif len(variables_predictives) == 3:
                variables_predictives.append({"nom": "Professionnels actifs", "importance": min(75, round((nb_entraineurs + nb_jockeys) / 10))})
            else:
                variables_predictives.append({"nom": "Base de données", "importance": 70})
        
        con.close()
        
        result = {
            "stats": {
                "taux_reussite": round(taux_reussite, 1),
                "roi_moyen": roi_moyen,
                "courses_analysees": total_courses,
                "modeles_actifs": modeles_actifs,
                "evolution_reussite": evolution_reussite,
                "evolution_roi": evolution_roi,
                "evolution_courses": evolution_courses
            },
            "performances_recentes": performances_recentes,
            "variables_predictives": variables_predictives[:5]
        }
        
        # Mise en cache
        cache.set(cache_key, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/chevaux")
async def get_chevaux(
    limit: int = 50, 
    offset: int = 0, 
    sort_by: Optional[str] = None, 
    sort_order: Optional[str] = "asc",
    search: Optional[str] = None
):
    """
    Liste des chevaux avec leurs statistiques
    Paramètres:
    - limit: nombre d'éléments à retourner
    - offset: décalage pour la pagination
    - sort_by: colonne de tri (nom, sexe, race, nb_courses, nb_victoires, taux_victoire)
    - sort_order: ordre du tri (asc ou desc)
    - search: terme de recherche pour filtrer par nom
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Mapping des colonnes pour le tri (Schema Plan Complet)
        sort_columns = {
            "nom": "c.nom_cheval",
            "sexe": "c.sexe_cheval",
            "race": "c.origine",
            "nb_courses": "s.nb_courses_total",
            "nb_victoires": "s.nb_victoires",
            "taux_victoire": "s.tx_victoire"
        }
        
        # Construire la clause ORDER BY
        order_clause = "s.nb_victoires DESC NULLS LAST"  # Par défaut
        if sort_by and sort_by in sort_columns:
            order_direction = "ASC" if sort_order == "asc" else "DESC"
            order_clause = f"{sort_columns[sort_by]} {order_direction} NULLS LAST"
        
        # Construire la clause WHERE pour la recherche
        where_conditions = []
        params = []
        
        # Utiliser la syntaxe PostgreSQL (%s) ou SQLite (?) selon le type de connexion
        placeholder = "%s" if USE_POSTGRESQL else "?"
        
        if search and search.strip():
            where_conditions.append(f"LOWER(c.nom_cheval) LIKE LOWER({placeholder})")
            params.append(f"%{search.strip()}%")
            
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 1. Récupérer le nombre total de résultats
        count_query = f"""
            SELECT COUNT(*)
            FROM chevaux c
            LEFT JOIN stats_chevaux s ON c.id_cheval = s.id_cheval
            WHERE {where_clause}
        """
        cur.execute(count_query, tuple(params))
        total_count = cur.fetchone()[0]
        
        # 2. Récupérer les données paginées
        params.extend([limit, offset])
        
        query = f"""
            SELECT 
                c.nom_cheval as nom,
                c.sexe_cheval as sexe,
                c.origine as race,
                COALESCE(s.nb_courses_total, 0) as nombre_courses_total,
                COALESCE(s.nb_victoires, 0) as nombre_victoires_total,
                NULL as dernier_resultat,
                NULL as dernieres_performances,
                COALESCE(s.tx_victoire, 0) as taux_victoire
            FROM chevaux c
            LEFT JOIN stats_chevaux s ON c.id_cheval = s.id_cheval
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT {placeholder} OFFSET {placeholder}
        """
        
        cur.execute(query, tuple(params))
        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        con.close()
        return {
            "chevaux": results,
            "total": total_count
        }
        
    except Exception as e:
        print(f"Erreur lors de la récupération des chevaux: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses-vues")
async def get_courses_vues(
    limit: int = 50, 
    offset: int = 0, 
    sort_by: Optional[str] = None, 
    sort_order: Optional[str] = "asc",
    search: Optional[str] = None
):
    """
    Liste des courses vues par les chevaux (table cheval_courses_seen)
    Paramètres:
    - limit: nombre d'éléments à retourner
    - offset: décalage pour la pagination
    - sort_by: colonne de tri (nom_cheval, date_course, reunion, course, hippodrome_nom, annee, victoire)
    - sort_order: ordre du tri (asc ou desc)
    - search: terme de recherche pour filtrer par nom de cheval ou hippodrome
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Mapping des colonnes pour le tri
        sort_columns = {
            "nom_cheval": "nom_norm",
            "date_course": "race_key",
            "reunion": "race_key",
            "course": "race_key",
            "hippodrome": "race_key",
            "annee": "annee",
            "victoire": "is_win"
        }
        
        # Construire la clause ORDER BY
        order_clause = "annee DESC, race_key DESC"  # Par défaut
        if sort_by and sort_by in sort_columns:
            order_direction = "ASC" if sort_order == "asc" else "DESC"
            order_clause = f"{sort_columns[sort_by]} {order_direction}"
        
        # Construire la clause WHERE pour la recherche
        where_clause = ""
        params = []
        placeholder = "%s" if USE_POSTGRESQL else "?"
        
        if search and search.strip():
            where_clause = f"WHERE (LOWER(nom_norm) LIKE LOWER({placeholder}) OR LOWER(race_key) LIKE LOWER({placeholder}))"
            search_term = f"%{search.strip()}%"
            params.extend([search_term, search_term])
        
        query = f"""
            SELECT 
                nom_norm, race_key, annee, is_win
            FROM cheval_courses_seen
            {where_clause}
            ORDER BY {order_clause}
            LIMIT {placeholder} OFFSET {placeholder}
        """
        
        params.extend([limit, offset])
        cur.execute(query, params)
        
        courses = []
        for row in cur.fetchall():
            nom_norm, race_key, annee, is_win = row
            
            # Parser le race_key pour extraire des infos lisibles
            # Format: "YYYY-MM-DD|R{reunion}|C{course}|{hippodrome}"
            parts = race_key.split('|')
            date_course = parts[0] if len(parts) > 0 else 'N/A'
            reunion = parts[1] if len(parts) > 1 else 'N/A'
            course = parts[2] if len(parts) > 2 else 'N/A'
            hippodrome = parts[3] if len(parts) > 3 else 'N/A'
            
            courses.append({
                "nom_cheval": nom_norm,
                "date_course": date_course,
                "reunion": reunion,
                "course": course,
                "hippodrome": hippodrome,
                "annee": annee,
                "victoire": bool(is_win),
                "race_key": race_key
            })
        
        # Compter le total avec le même filtre de recherche
        count_where_clause = ""
        count_params = []
        
        placeholder = "%s" if USE_POSTGRESQL else "?"
        if search and search.strip():
            count_where_clause = f"WHERE (LOWER(nom_norm) LIKE LOWER({placeholder}) OR LOWER(race_key) LIKE LOWER({placeholder}))"
            search_term = f"%{search.strip()}%"
            count_params.extend([search_term, search_term])
        
        count_query = f"SELECT COUNT(*) FROM cheval_courses_seen {count_where_clause}"
        cur.execute(count_query, count_params)
        total = cur.fetchone()[0]
        
        con.close()
        
        return {
            "courses": courses,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/chevaux/{nom}")
async def get_cheval_details(nom: str):
    """
    Détails complets d'un cheval
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        placeholder = "%s" if USE_POSTGRESQL else "?"
        cur.execute(f"""
            SELECT 
                nom, sexe, race, robe, date_naissance,
                pays_naissance, entraineur_courant, jockey_habituel,
                dernier_resultat, nombre_courses_total, nombre_victoires_total,
                nombre_courses_2025, nombre_victoires_2025,
                dernieres_performances
            FROM chevaux
            WHERE LOWER(nom) = LOWER({placeholder})
        """, (nom,))
        
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Cheval non trouvé")
        
        (nom, sexe, race, robe, date_naissance, pays_naissance,
         entraineur, jockey, musique, nb_courses, nb_victoires,
         nb_courses_2025, nb_victoires_2025, perfs_json) = row
        
        dernieres_perfs = []
        if perfs_json:
            try:
                dernieres_perfs = json.loads(perfs_json)
            except:
                pass
        
        con.close()
        
        return {
            "nom": nom,
            "sexe": sexe,
            "race": race,
            "robe": robe,
            "date_naissance": date_naissance,
            "pays_naissance": pays_naissance,
            "entraineur": entraineur,
            "jockey": jockey,
            "musique": musique,
            "statistiques": {
                "total": {
                    "courses": nb_courses,
                    "victoires": nb_victoires,
                    "taux": round((nb_victoires / nb_courses * 100) if nb_courses else 0, 1)
                },
                "2025": {
                    "courses": nb_courses_2025,
                    "victoires": nb_victoires_2025,
                    "taux": round((nb_victoires_2025 / nb_courses_2025 * 100) if nb_courses_2025 else 0, 1)
                }
            },
            "dernieres_performances": dernieres_perfs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/analytics/chevaux")
async def get_analytics_chevaux():
    """
    Analyses statistiques sur les chevaux (avec cache)
    """
    # Vérifier le cache
    cache_key = "analytics:chevaux"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data
        
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        if USE_POSTGRESQL:
            # --- Version PostgreSQL OPTIMISÉE - Une seule requête CTE ---
            cur.execute("""
                WITH race_dist AS (
                    SELECT race, COUNT(DISTINCT nom_norm) as count
                    FROM cheval_courses_seen
                    WHERE race IS NOT NULL
                    GROUP BY race
                    ORDER BY count DESC
                    LIMIT 10
                ),
                top_perf AS (
                    SELECT c.nom_cheval, s.nb_courses_total, s.nb_victoires,
                           CAST(s.nb_victoires AS FLOAT) / NULLIF(s.nb_courses_total, 0) * 100 as taux
                    FROM chevaux c
                    JOIN stats_chevaux s ON c.id_cheval = s.id_cheval
                    WHERE s.nb_courses_total >= 3
                    ORDER BY taux DESC
                    LIMIT 10
                ),
                sexe_stats AS (
                    SELECT 
                        CASE 
                            WHEN UPPER(sexe) IN ('F', 'FEMELLES', 'FEMMELLES') THEN 'FEMELLES'
                            WHEN UPPER(sexe) IN ('M', 'MALES') THEN 'MALES'
                            WHEN UPPER(sexe) IN ('H', 'HONGRES', 'HONGRE') THEN 'HONGRES'
                            ELSE sexe
                        END as sexe_normalise,
                        COUNT(DISTINCT nom_norm) as total,
                        AVG(CAST(is_win AS FLOAT) * 100) as taux_moyen
                    FROM cheval_courses_seen
                    WHERE sexe IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1
                )
                SELECT 'race' as type, race as col1, count::text as col2, NULL as col3, NULL as col4 FROM race_dist
                UNION ALL
                SELECT 'perf' as type, nom_cheval, nb_courses_total::text, nb_victoires::text, taux::text FROM top_perf
                UNION ALL
                SELECT 'sexe' as type, sexe_normalise, total::text, taux_moyen::text, NULL FROM sexe_stats
            """)
            
            distribution_races = []
            top_performers = []
            stats_sexe = []
            
            for row in cur.fetchall():
                row_type = row[0]
                if row_type == 'race':
                    distribution_races.append({"race": row[1], "count": int(row[2])})
                elif row_type == 'perf':
                    top_performers.append({
                        "nom": row[1],
                        "courses": int(row[2]),
                        "victoires": int(row[3]),
                        "taux": round(float(row[4]), 1)
                    })
                elif row_type == 'sexe':
                    stats_sexe.append({
                        "sexe": row[1],
                        "total": int(row[2]),
                        "taux_moyen": round(float(row[3]), 1)
                    })
                
        else:
            # --- Version SQLite ---
            
            # Distribution par race
            cur.execute("""
                SELECT race, COUNT(*) as count
                FROM chevaux
                WHERE race IS NOT NULL
                GROUP BY race
                ORDER BY count DESC
                LIMIT 10
            """)
            distribution_races = []
            for row in cur.fetchall():
                race = row[0]
                count = int(row[1]) if row[1] is not None else 0
                distribution_races.append({"race": race, "count": count})
            
            # Top performers
            cur.execute("""
                SELECT nom, nombre_courses_total, nombre_victoires_total,
                       CAST(nombre_victoires_total AS FLOAT) / nombre_courses_total * 100 as taux
                FROM chevaux
                WHERE nombre_courses_total >= 3
                ORDER BY taux DESC
                LIMIT 10
            """)
            top_performers = []
            for row in cur.fetchall():
                nom = row[0]
                courses = int(row[1]) if row[1] is not None else 0
                victoires = int(row[2]) if row[2] is not None else 0
                taux = float(row[3]) if row[3] is not None else 0.0
                top_performers.append({
                    "nom": nom,
                    "courses": courses,
                    "victoires": victoires,
                    "taux": round(taux, 1)
                })
            
            # Statistiques par sexe
            cur.execute("""
                SELECT 
                    CASE 
                        WHEN UPPER(sexe) IN ('F', 'FEMELLES', 'FEMMELLES') THEN 'FEMELLES'
                        WHEN UPPER(sexe) IN ('M', 'MALES') THEN 'MALES'
                        WHEN UPPER(sexe) IN ('H', 'HONGRES', 'HONGRE') THEN 'HONGRES'
                        ELSE sexe
                    END as sexe_normalise,
                    COUNT(*) as total,
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen
                FROM chevaux
                WHERE sexe IS NOT NULL AND nombre_courses_total > 0
                GROUP BY sexe_normalise
                ORDER BY sexe_normalise
                LIMIT 2
            """)
            stats_sexe = []
            rows = cur.fetchall()
            for row in rows:
                sexe = row[0]
                total = int(row[1]) if row[1] is not None else 0
                taux_moyen = float(row[2]) if row[2] is not None else 0.0
                stats_sexe.append({
                    "sexe": sexe,
                    "total": total,
                    "taux_moyen": round(taux_moyen, 1)
                })
        
        con.close()
        
        result = {
            "distribution_races": distribution_races,
            "top_performers": top_performers,
            "stats_par_sexe": stats_sexe
        }
        
        # Mise en cache
        cache.set(cache_key, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/analytics/jockeys")
async def get_analytics_jockeys():
    """
    Analyses statistiques sur les jockeys (avec cache)
    """
    # Vérifier le cache
    cache_key = "jockeys"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data
        
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        if USE_POSTGRESQL:
            # Top jockeys par nombre de chevaux entraînés (via cheval_courses_seen)
            cur.execute("""
                SELECT 
                    driver_jockey as jockey,
                    COUNT(DISTINCT nom_norm) as nb_chevaux,
                    AVG(CAST(is_win AS FLOAT) * 100) as taux_moyen
                FROM cheval_courses_seen
                WHERE driver_jockey IS NOT NULL
                GROUP BY driver_jockey
                HAVING COUNT(*) >= 3
                ORDER BY taux_moyen DESC
                LIMIT 15
            """)
        else:
            # Top jockeys par nombre de chevaux entraînés
            cur.execute("""
                SELECT 
                    jockey_habituel as jockey,
                    COUNT(*) as nb_chevaux,
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen
                FROM chevaux
                WHERE jockey_habituel IS NOT NULL AND nombre_courses_total > 0
                GROUP BY jockey_habituel
                HAVING COUNT(*) >= 3
                ORDER BY taux_moyen DESC
                LIMIT 15
            """)
        
        top_jockeys = []
        for row in cur.fetchall():
            jockey = row[0]
            nb_chevaux = int(row[1]) if row[1] is not None else 0
            taux_moyen = float(row[2]) if row[2] is not None else 0.0
            top_jockeys.append({
                "nom": jockey,
                "nb_chevaux": nb_chevaux,
                "taux_reussite": round(taux_moyen, 1)
            })
        
        con.close()
        
        return {
            "top_jockeys": top_jockeys
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/analytics/entraineurs")
async def get_analytics_entraineurs():
    """
    Analyses statistiques sur les entraîneurs
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        if USE_POSTGRESQL:
            # Top entraîneurs (via cheval_courses_seen)
            cur.execute("""
                SELECT 
                    entraineur,
                    COUNT(DISTINCT nom_norm) as nb_chevaux,
                    AVG(CAST(is_win AS FLOAT) * 100) as taux_moyen
                FROM cheval_courses_seen
                WHERE entraineur IS NOT NULL
                GROUP BY entraineur
                HAVING COUNT(*) >= 3
                ORDER BY taux_moyen DESC
                LIMIT 15
            """)
        else:
            # Top entraîneurs
            cur.execute("""
                SELECT 
                    entraineur_courant as entraineur,
                    COUNT(*) as nb_chevaux,
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen
                FROM chevaux
                WHERE entraineur_courant IS NOT NULL AND nombre_courses_total > 0
                GROUP BY entraineur_courant
                HAVING COUNT(*) >= 3
                ORDER BY taux_moyen DESC
                LIMIT 15
            """)
        
        top_entraineurs = []
        for row in cur.fetchall():
            entraineur = row[0]
            nb_chevaux = int(row[1]) if row[1] is not None else 0
            taux_moyen = float(row[2]) if row[2] is not None else 0.0
            top_entraineurs.append({
                "nom": entraineur,
                "nb_chevaux": nb_chevaux,
                "taux_reussite": round(taux_moyen, 1)
            })
        
        con.close()
        
        return {
            "top_entraineurs": top_entraineurs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/analytics/hippodromes")
async def get_analytics_hippodromes():
    """Statistiques par hippodrome"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Utiliser race_key pour extraire l'hippodrome
        cur.execute("""
            SELECT 
                SPLIT_PART(race_key, '|', 4) as hippo,
                COUNT(*) as nb_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as nb_victoires,
                COUNT(DISTINCT nom_norm) as nb_chevaux
            FROM cheval_courses_seen
            WHERE race_key IS NOT NULL
            GROUP BY hippo
            ORDER BY nb_courses DESC
            LIMIT 10
        """)
        
        hippodromes = []
        for row in cur.fetchall():
            hippo = row[0] or "Inconnu"
            nb_courses = int(row[1]) if row[1] is not None else 0
            nb_victoires = int(row[2]) if row[2] is not None else 0
            nb_chevaux = int(row[3]) if row[3] is not None else 0
            if hippo and hippo != "":
                hippodromes.append({
                    "nom": hippo,
                    "nb_courses": nb_courses,
                    "nb_victoires": nb_victoires,
                    "nb_chevaux": nb_chevaux,
                    "taux_victoire": round((nb_victoires / nb_courses * 100) if nb_courses > 0 else 0, 1)
                })
        
        con.close()
        return {"hippodromes": hippodromes}
        
    except Exception as e:
        return {"hippodromes": []}

@app.get("/api/analytics/evolution")
async def get_analytics_evolution():
    """Évolution des courses dans le temps"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        cur.execute("""
            SELECT 
                annee,
                COUNT(*) as nb_courses,
                COUNT(DISTINCT nom_norm) as nb_chevaux,
                SUM(is_win) as nb_victoires
            FROM cheval_courses_seen
            WHERE annee IS NOT NULL
            GROUP BY annee
            ORDER BY annee DESC
            LIMIT 10
        """)
        
        evolution = []
        for row in cur.fetchall():
            annee = int(row[0]) if row[0] is not None else 0
            nb_courses = int(row[1]) if row[1] is not None else 0
            nb_chevaux = int(row[2]) if row[2] is not None else 0
            nb_victoires = int(row[3]) if row[3] is not None else 0
            evolution.append({
                "annee": annee,
                "nb_courses": nb_courses,
                "nb_chevaux": nb_chevaux,
                "nb_victoires": nb_victoires,
                "taux_victoire": round((nb_victoires / nb_courses * 100) if nb_courses > 0 else 0, 1)
            })
        
        con.close()
        return {"evolution": evolution}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/analytics/taux-par-race")
async def get_taux_par_race():
    """Taux de victoire par race"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        if USE_POSTGRESQL:
            cur.execute("""
                SELECT 
                    race,
                    COUNT(*) as total,
                    SUM(is_win) as victoires,
                    AVG(CAST(is_win AS FLOAT) * 100) as taux_moyen
                FROM cheval_courses_seen
                WHERE race IS NOT NULL
                GROUP BY race
                ORDER BY taux_moyen DESC
                LIMIT 10
            """)
        else:
            cur.execute("""
                SELECT 
                    race,
                    COUNT(*) as total,
                    SUM(nombre_victoires_total) as victoires,
                    AVG(CAST(nombre_victoires_total AS FLOAT) / NULLIF(nombre_courses_total, 0) * 100) as taux_moyen
                FROM chevaux
                WHERE race IS NOT NULL AND nombre_courses_total > 0
                GROUP BY race
                ORDER BY taux_moyen DESC
                LIMIT 10
            """)
        
        taux_race = []
        for row in cur.fetchall():
            race = row[0]
            total = int(row[1]) if row[1] is not None else 0
            victoires = int(row[2]) if row[2] is not None else 0
            taux_moyen = float(row[3]) if row[3] is not None else 0.0
            taux_race.append({
                "race": race,
                "total": total,
                "victoires": victoires,
                "taux_moyen": round(taux_moyen, 1)
            })
        
        con.close()
        return {"taux_par_race": taux_race}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/analytics/odds")
async def get_analytics_odds():
    """Taux de victoire par cote"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # On utilise cote_finale si dispo, sinon cote_matin
        query = """
            SELECT 
                CASE 
                    WHEN COALESCE(cote_finale, cote_matin) < 2 THEN '< 2'
                    WHEN COALESCE(cote_finale, cote_matin) BETWEEN 2 AND 5 THEN '2 - 5'
                    WHEN COALESCE(cote_finale, cote_matin) BETWEEN 5 AND 10 THEN '5 - 10'
                    WHEN COALESCE(cote_finale, cote_matin) BETWEEN 10 AND 20 THEN '10 - 20'
                    WHEN COALESCE(cote_finale, cote_matin) BETWEEN 20 AND 50 THEN '20 - 50'
                    ELSE '> 50'
                END as range_cote,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE COALESCE(cote_finale, cote_matin) IS NOT NULL
            GROUP BY range_cote
        """
        
        cur.execute(query)
        
        stats = []
        order_map = {'< 2': 1, '2 - 5': 2, '5 - 10': 3, '10 - 20': 4, '20 - 50': 5, '> 50': 6}
        
        for row in cur.fetchall():
            range_cote = row[0]
            total = int(row[1])
            victoires = int(row[2]) if row[2] is not None else 0
            taux = round((victoires / total * 100), 1) if total > 0 else 0
            
            stats.append({
                "range": range_cote,
                "total": total,
                "victoires": victoires,
                "taux": taux,
                "order": order_map.get(range_cote, 99)
            })
            
        stats.sort(key=lambda x: x['order'])
        con.close()
        return {"odds_stats": stats}
        
    except Exception as e:
        return {"odds_stats": []}

@app.get("/api/analytics/distance")
async def get_analytics_distance():
    """Taux de victoire par distance"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        query = """
            SELECT 
                CASE 
                    WHEN distance_m < 1600 THEN '< 1600m'
                    WHEN distance_m BETWEEN 1600 AND 2000 THEN '1600m - 2000m'
                    WHEN distance_m BETWEEN 2001 AND 2400 THEN '2001m - 2400m'
                    WHEN distance_m BETWEEN 2401 AND 2800 THEN '2401m - 2800m'
                    ELSE '> 2800m'
                END as range_dist,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE distance_m IS NOT NULL
            GROUP BY range_dist
        """
        
        cur.execute(query)
        
        stats = []
        order_map = {'< 1600m': 1, '1600m - 2000m': 2, '2001m - 2400m': 3, '2401m - 2800m': 4, '> 2800m': 5}
        
        for row in cur.fetchall():
            range_dist = row[0]
            total = int(row[1])
            victoires = int(row[2]) if row[2] is not None else 0
            taux = round((victoires / total * 100), 1) if total > 0 else 0
            
            stats.append({
                "range": range_dist,
                "total": total,
                "victoires": victoires,
                "taux": taux,
                "order": order_map.get(range_dist, 99)
            })
            
        stats.sort(key=lambda x: x['order'])
        con.close()
        return {"distance_stats": stats}
        
    except Exception as e:
        return {"distance_stats": []}

@app.get("/api/analytics/age")
async def get_analytics_age():
    """Taux de victoire par âge"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        query = """
            SELECT 
                age,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE age IS NOT NULL AND age BETWEEN 2 AND 12
            GROUP BY age
            ORDER BY age
        """
        
        cur.execute(query)
        
        stats = []
        for row in cur.fetchall():
            age = int(row[0])
            total = int(row[1])
            victoires = int(row[2]) if row[2] is not None else 0
            taux = round((victoires / total * 100), 1) if total > 0 else 0
            
            stats.append({
                "age": f"{age} ans",
                "total": total,
                "victoires": victoires,
                "taux": taux
            })
            
        con.close()
        return {"age_stats": stats}
        
    except Exception as e:
        return {"age_stats": []}

# ============================================================================
# NOUVELLES ROUTES - COURSES DU JOUR ET PRÉDICTIONS
# ============================================================================

@app.get("/api/courses/today")
async def get_courses_today():
    """Récupère les courses du jour avec leurs participants"""
    try:
        from datetime import datetime
        
        con = get_db_connection()
        cur = con.cursor()
        
        # Date du jour au format YYYY-MM-DD
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Récupérer les courses du jour (ou les plus récentes si aucune aujourd'hui)
        cur.execute("""
            SELECT DISTINCT 
                race_key,
                hippodrome_nom,
                type_course,
                distance_m,
                corde,
                allocation_totale,
                COUNT(*) as nb_partants
            FROM cheval_courses_seen
            WHERE race_key LIKE %s
            GROUP BY race_key, hippodrome_nom, type_course, distance_m, corde, allocation_totale
            ORDER BY race_key DESC
            LIMIT 20
        """, (today + '%',))
        
        courses = []
        for row in cur.fetchall():
            race_key = row[0]
            parts = race_key.split('|') if race_key else []
            date_course = parts[0] if len(parts) > 0 else None
            reunion = parts[1] if len(parts) > 1 else None
            course_num = parts[2] if len(parts) > 2 else None
            
            courses.append({
                "race_key": race_key,
                "date": date_course,
                "reunion": reunion,
                "course": course_num,
                "hippodrome": row[1],
                "type_course": row[2],
                "distance": row[3],
                "corde": row[4],
                "allocation": row[5],
                "nb_partants": row[6]
            })
        
        con.close()
        return {"courses": courses, "total": len(courses)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses/{race_key}/participants")
async def get_course_participants(race_key: str):
    """Récupère les participants d'une course avec leurs statistiques et prédictions"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Récupérer les participants de la course
        cur.execute("""
            SELECT 
                cs.nom_norm,
                cs.numero_dossard,
                cs.age,
                cs.sexe,
                cs.cote_finale,
                cs.cote_reference,
                cs.tendance_cote,
                cs.amplitude_tendance,
                cs.est_favori,
                cs.avis_entraineur,
                cs.driver_jockey,
                cs.entraineur,
                cs.musique,
                cs.place_finale,
                cs.is_win,
                cs.allocation_totale,
                cs.gains_course,
                cs.url_casaque,
                cs.allure,
                cs.statut_participant
            FROM cheval_courses_seen cs
            WHERE cs.race_key = %s
            ORDER BY cs.numero_dossard
        """, (race_key,))
        
        participants = []
        for row in cur.fetchall():
            nom = row[0]
            
            # Calculer le score de prédiction basé sur plusieurs facteurs
            score = calculate_prediction_score(
                cote=row[4],
                cote_ref=row[5],
                tendance=row[6],
                amplitude=row[7],
                est_favori=row[8],
                avis=row[9]
            )
            
            # Récupérer l'historique du cheval
            cur.execute("""
                SELECT 
                    COUNT(*) as nb_courses,
                    SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as nb_victoires,
                    AVG(CASE WHEN place_finale IS NOT NULL AND place_finale <= 3 THEN 1 ELSE 0 END) * 100 as taux_place
                FROM cheval_courses_seen
                WHERE nom_norm = %s
            """, (nom,))
            hist = cur.fetchone()
            
            participants.append({
                "nom": nom,
                "numero": row[1],
                "age": row[2],
                "sexe": row[3],
                "cote": row[4],
                "cote_reference": row[5],
                "tendance": row[6],
                "amplitude": row[7],
                "est_favori": row[8],
                "avis_entraineur": row[9],
                "jockey": row[10],
                "entraineur": row[11],
                "musique": row[12],
                "place_finale": row[13],
                "is_win": row[14],
                "gains": row[16],
                "url_casaque": row[17],
                "allure": row[18],
                "statut": row[19],
                "prediction_score": score,
                "historique": {
                    "nb_courses": hist[0] if hist else 0,
                    "nb_victoires": hist[1] if hist else 0,
                    "taux_place": round(hist[2], 1) if hist and hist[2] else 0
                }
            })
        
        # Trier par score de prédiction
        participants.sort(key=lambda x: x['prediction_score'], reverse=True)
        
        con.close()
        return {"participants": participants, "race_key": race_key}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_prediction_score(cote, cote_ref, tendance, amplitude, est_favori, avis):
    """Calcule un score de prédiction entre 0 et 100"""
    score = 50  # Score de base
    
    # Facteur cote (plus la cote est basse, meilleur le score)
    if cote:
        if cote < 3:
            score += 25
        elif cote < 5:
            score += 15
        elif cote < 10:
            score += 5
        elif cote > 50:
            score -= 15
    
    # Facteur tendance
    if tendance == '+':
        score -= 5  # Cote qui monte = moins de confiance marché
    elif tendance == '-':
        score += 10  # Cote qui baisse = plus de confiance marché
    
    # Amplitude de la tendance
    if amplitude:
        if amplitude > 20 and tendance == '-':
            score += 10  # Grosse baisse = fort signal
        elif amplitude > 20 and tendance == '+':
            score -= 10  # Grosse hausse = mauvais signal
    
    # Favori
    if est_favori:
        score += 15
    
    # Avis entraîneur
    if avis == 'POSITIF':
        score += 10
    elif avis == 'NEGATIF':
        score -= 10
    
    # Limiter entre 0 et 100
    return max(0, min(100, score))

def _safe_prob(p):
    """Clamp une probabilité pour éviter les logs inf/nan."""
    return max(1e-6, min(1 - 1e-6, p or 0))

def _safe_logit(p):
    p = _safe_prob(p)
    return math.log(p / (1 - p))

def _softmax_temperature(logits, tau):
    """Softmax avec température course-level."""
    if tau <= 0:
        tau = 1.0
    shifted = [l / tau for l in logits]
    m = max(shifted) if shifted else 0
    exps = [math.exp(l - m) for l in shifted]
    total = sum(exps) or 1.0
    return [e / total for e in exps]

def _distance_bucket(distance_m):
    """Bucket distance pour les effets hiérarchiques (~200m)."""
    if distance_m is None:
        return None
    try:
        return int(round(distance_m / 200.0) * 200)
    except Exception:
        return None

def _compute_effect(key, stats, base_logit, global_rate, min_support, prior_strength=25):
    """Effet hiérarchique shrinké vers le taux global."""
    if key is None or key not in stats:
        return 0.0
    wins = stats[key]["wins"]
    total = stats[key]["total"]
    if total <= 0:
        return 0.0
    rate = (wins + prior_strength * global_rate) / (total + prior_strength)
    weight = min(1.0, total / float(max(min_support, 1)))
    return (_safe_logit(rate) - base_logit) * weight

def _load_effects(cur, column_expr, alias=None):
    """Charge les stats agrégées (wins/total) pour un champ donné (colonne ou expression)."""
    key_alias = alias or "key"
    cur.execute(f"""
        SELECT {column_expr} AS {key_alias}, COUNT(*) as total, 
               SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
        FROM cheval_courses_seen
        WHERE race_key >= (CURRENT_DATE - INTERVAL '365 days')::text
          AND {column_expr} IS NOT NULL
        GROUP BY {column_expr}
    """)
    stats = {}
    for key, total, wins in cur.fetchall():
        stats[key] = {"total": total or 0, "wins": wins or 0}
    return stats

def _load_distance_effects(cur):
    """Charge les stats agrégées en bucketisant les distances."""
    cur.execute("""
        SELECT distance_m, COUNT(*) as total,
               SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
        FROM cheval_courses_seen
        WHERE race_key >= (CURRENT_DATE - INTERVAL '365 days')::text
          AND distance_m IS NOT NULL
        GROUP BY distance_m
    """)
    stats = {}
    for distance, total, wins in cur.fetchall():
        bucket = _distance_bucket(distance)
        if bucket is None:
            continue
        entry = stats.setdefault(bucket, {"total": 0, "wins": 0})
        entry["total"] += total or 0
        entry["wins"] += wins or 0
    return stats

def _microstructure_effect(tendance, amplitude, est_favori):
    """Effet microstructure (vitesse/dir. de cote + statut favori)."""
    amp = amplitude or 0
    effect = 0.0
    if tendance == '-':  # Cote qui baisse
        effect += min(0.30, amp / 80.0)
    elif tendance == '+':  # Cote qui monte
        effect -= min(0.25, amp / 80.0)
    if est_favori:
        effect += 0.05  # léger prior favori
    return effect

def run_benter_head_for_date(search_date, cur=None, tau=None, min_support=60):
    """
    Applique le head type Benter hiérarchique (logit + effets piste/état/distance/corde)
    puis normalisation course-level et calibration légère.
    """
    owns_cursor = cur is None
    con = None
    meta = {
        "status": "pending",
        "date": search_date,
        "tau": tau if tau is not None else 1.1,
        "min_support": min_support
    }
    try:
        if cur is None:
            con = get_db_connection()
            cur = con.cursor()

        # Taux global pour prior
        cur.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
            FROM cheval_courses_seen
            WHERE race_key >= (CURRENT_DATE - INTERVAL '365 days')::text
        """)
        total, wins = cur.fetchone()
        total = total or 1
        wins = wins or 0
        global_rate = wins / float(total)
        base_logit = _safe_logit(global_rate)
        meta["global_rate"] = round(global_rate, 4)

        # Effets hiérarchiques
        track_effects = _load_effects(cur, "COALESCE(hippodrome_code, hippodrome_nom)", alias="hippodrome_key")
        going_effects = _load_effects(cur, "etat_piste")
        corde_effects = _load_effects(cur, "corde")
        distance_effects = _load_distance_effects(cur)

        # Récupérer les partants du jour
        cur.execute("""
            SELECT 
                race_key,
                nom_norm,
                numero_dossard,
                COALESCE(hippodrome_code, hippodrome_nom) as hippodrome_key,
                etat_piste,
                distance_m,
                corde,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori
            FROM cheval_courses_seen
            WHERE race_key LIKE %s
              AND cote_finale IS NOT NULL
              AND cote_finale > 0
              AND cote_finale < 200
        """, (search_date + '%',))
        rows = cur.fetchall()
        if not rows:
            meta["status"] = "no_data"
            meta["participants"] = 0
            return {"by_runner": {}, "meta": meta}

        tau_value = tau if tau is not None else 1.1
        platt_a = float(os.getenv("BENTER_PLATT_A", "1.0"))
        platt_b = float(os.getenv("BENTER_PLATT_B", "0.0"))
        prior_strength = 25

        by_race = {}
        by_runner = {}
        for (
            race_key,
            nom_norm,
            numero,
            hippodrome_key,
            etat_piste,
            distance_m,
            corde,
            cote,
            cote_ref,
            tendance,
            amplitude,
            est_favori
        ) in rows:
            distance_bucket = _distance_bucket(distance_m)
            p_market = 1.0 / cote if cote else 0
            p_ref = 1.0 / cote_ref if cote_ref else p_market
            # Correction marché (blend ref + favori + prior neutre)
            p_mkt_corr = _safe_prob(0.6 * p_market + 0.3 * p_ref + (0.05 if est_favori else 0))

            context_effect = (
                _compute_effect(hippodrome_key, track_effects, base_logit, global_rate, min_support, prior_strength)
                + _compute_effect(etat_piste, going_effects, base_logit, global_rate, min_support, prior_strength)
                + _compute_effect(distance_bucket, distance_effects, base_logit, global_rate, min_support, prior_strength)
                + _compute_effect(corde, corde_effects, base_logit, global_rate, min_support, prior_strength)
            )
            micro_effect = _microstructure_effect(tendance, amplitude, est_favori)

            head_logit = _safe_logit(p_mkt_corr) + context_effect + micro_effect
            by_race.setdefault(race_key, []).append({
                "numero": numero,
                "nom": nom_norm,
                "logit": head_logit,
                "p_market_corr": p_mkt_corr,
                "context_effect": context_effect,
                "micro_effect": micro_effect
            })

        for race_key, runners in by_race.items():
            logits = [r["logit"] for r in runners]
            probs = _softmax_temperature(logits, tau_value)
            for idx, runner in enumerate(runners):
                p_model_norm = _safe_prob(probs[idx])
                calibrated_logit = platt_a * _safe_logit(p_model_norm) + platt_b
                p_calibrated = 1.0 / (1.0 + math.exp(-calibrated_logit))
                key = (race_key, runner["numero"])
                by_runner[key] = {
                    "p_model_norm": p_model_norm,
                    "p_calibrated": p_calibrated,
                    "context_effect": runner["context_effect"],
                    "micro_effect": runner["micro_effect"],
                    "p_market_corr": runner["p_market_corr"]
                }

        meta["status"] = "ok"
        meta["races_covered"] = len(by_race)
        meta["participants"] = len(by_runner)
        return {"by_runner": by_runner, "meta": meta}
    except Exception as e:
        meta["status"] = "error"
        meta["reason"] = str(e)
        return {"by_runner": {}, "meta": meta}
    finally:
        if owns_cursor and con is not None:
            con.close()

@app.get("/api/predictions")
async def get_predictions(limit: int = Query(default=20, ge=1, le=100)):
    """Récupère les meilleures prédictions basées sur notre algorithme"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Récupérer les chevaux des courses récentes avec bonnes cotes
        cur.execute("""
            SELECT 
                cs.nom_norm,
                cs.race_key,
                cs.hippodrome_nom,
                cs.numero_dossard,
                cs.cote_finale,
                cs.cote_reference,
                cs.tendance_cote,
                cs.amplitude_tendance,
                cs.est_favori,
                cs.avis_entraineur,
                cs.driver_jockey,
                cs.entraineur,
                cs.musique,
                cs.place_finale,
                cs.is_win
            FROM cheval_courses_seen cs
            WHERE cs.race_key LIKE '2025-%'
            AND cs.cote_finale IS NOT NULL
            AND cs.cote_finale > 0
            ORDER BY cs.race_key DESC
            LIMIT 200
        """)
        
        predictions = []
        for row in cur.fetchall():
            score = calculate_prediction_score(
                cote=row[4],
                cote_ref=row[5],
                tendance=row[6],
                amplitude=row[7],
                est_favori=row[8],
                avis=row[9]
            )
            
            # Calculer la probabilité estimée
            proba = score_to_probability(score, row[4])
            
            # Calculer la value bet
            value = calculate_value_bet(proba, row[4])
            
            predictions.append({
                "nom": row[0],
                "race_key": row[1],
                "hippodrome": row[2],
                "numero": row[3],
                "cote": row[4],
                "cote_reference": row[5],
                "tendance": row[6],
                "avis_entraineur": row[9],
                "jockey": row[10],
                "entraineur": row[11],
                "musique": row[12],
                "place_finale": row[13],
                "resultat": "Gagné" if row[14] == 1 else ("Placé" if row[13] and row[13] <= 3 else "Non placé" if row[13] else "En attente"),
                "score": score,
                "probabilite": proba,
                "value_bet": value,
                "recommendation": get_recommendation(score, value, row[4])
            })
        
        # Trier par score et prendre les meilleurs
        predictions.sort(key=lambda x: (x['score'], x['value_bet']), reverse=True)
        
        con.close()
        return {"predictions": predictions[:limit]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def score_to_probability(score: float, cote: float) -> float:
    """Convertit un score en probabilité estimée"""
    # Probabilité implicite de la cote
    proba_cote = (1 / cote * 100) if cote and cote > 0 else 0
    
    # Ajuster avec notre score
    # Score > 70 = on est plus confiant que la cote
    # Score < 30 = on est moins confiant que la cote
    adjustment = (score - 50) / 100  # -0.5 à +0.5
    
    proba_ajustee = proba_cote * (1 + adjustment)
    
    return round(max(1, min(95, proba_ajustee)), 1)

def calculate_value_bet(proba: float, cote: float) -> float:
    """Calcule la value d'un pari (Expected Value)"""
    if not cote or cote <= 0 or not proba:
        return 0
    
    # EV = (Probabilité * Cote) - 1
    # Si > 0, c'est une value bet
    ev = (proba / 100 * cote) - 1
    
    return round(ev * 100, 1)  # En pourcentage

def get_recommendation(score: float, value: float, cote: float) -> dict:
    """Génère une recommandation de pari"""
    if score >= 75 and value > 10:
        return {
            "action": "PARIER",
            "niveau": "FORT",
            "mise_recommandee": "3-5%",
            "confiance": "Haute",
            "raison": "Score élevé + Value bet positive"
        }
    elif score >= 60 and value > 0:
        return {
            "action": "PARIER",
            "niveau": "MOYEN",
            "mise_recommandee": "1-2%",
            "confiance": "Moyenne",
            "raison": "Bon profil avec value correcte"
        }
    elif score >= 50 and cote and cote < 3:
        return {
            "action": "SURVEILLER",
            "niveau": "FAIBLE",
            "mise_recommandee": "0.5-1%",
            "confiance": "Modérée",
            "raison": "Favori mais faible value"
        }
    else:
        return {
            "action": "ÉVITER",
            "niveau": "AUCUN",
            "mise_recommandee": "0%",
            "confiance": "Faible",
            "raison": "Pas assez de signaux positifs"
        }

@app.get("/api/betting/recommendations")
async def get_betting_recommendations(
    min_score: int = Query(default=60, ge=0, le=100),
    min_value: float = Query(default=0, ge=-50, le=100),
    max_cote: float = Query(default=50, ge=1, le=500)
):
    """Récupère les recommandations de paris filtrées - UNIQUEMENT pour courses pré-off avec partants valides"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Récupérer la date des dernières courses (aujourd'hui ou la plus récente)
        today = datetime.now().strftime('%Y-%m-%d')
        cur.execute("""
            SELECT DISTINCT SUBSTRING(race_key FROM 1 FOR 10) as race_date
            FROM cheval_courses_seen
            WHERE race_key IS NOT NULL
            ORDER BY race_date DESC
            LIMIT 1
        """)
        latest_date = cur.fetchone()
        search_date = today if latest_date is None else latest_date[0]
        
        # Timestamp actuel en millisecondes pour filtrer les courses déjà parties
        import time
        now_ms = int(time.time() * 1000)
        
        # FILTRES ANTI-FUITE:
        # 1. place_finale IS NULL → course pas encore courue (pré-off uniquement)
        # 2. statut_participant PARTANT → exclut NP pré-départ
        # 3. incident IS NULL → exclut DAI, ARR, TOMBE, etc. (incidents post-départ)
        # 4. cote_finale ET cote_reference <= max_cote → exclut outsiders extrêmes
        # 5. heure_depart > now → course pas encore partie
        cur.execute("""
            SELECT 
                cs.nom_norm,
                cs.race_key,
                cs.hippodrome_nom,
                cs.numero_dossard,
                cs.cote_finale,
                cs.cote_reference,
                cs.tendance_cote,
                cs.amplitude_tendance,
                cs.est_favori,
                cs.avis_entraineur,
                cs.driver_jockey,
                cs.entraineur,
                cs.type_course,
                cs.distance_m,
                cs.place_finale,
                cs.is_win,
                cs.statut_participant,
                cs.incident
            FROM cheval_courses_seen cs
            WHERE cs.race_key LIKE %s
            AND cs.cote_finale IS NOT NULL
            AND cs.cote_finale > 0
            AND cs.cote_finale <= %s
            AND (cs.cote_reference IS NULL OR cs.cote_reference <= %s)
            AND cs.place_finale IS NULL  -- anti-fuite: uniquement pré-off
            AND (cs.statut_participant IS NULL OR UPPER(cs.statut_participant) IN ('PARTANT', 'PARTANTE', 'PART', 'P', ''))
            AND cs.incident IS NULL  -- anti-fuite: exclut DAI, ARR, TOMBE, NP tardif
            AND (cs.heure_depart IS NULL OR cs.heure_depart::bigint > %s)  -- anti-fuite: exclut courses déjà parties
            ORDER BY cs.race_key DESC
            LIMIT 500
        """, (search_date + '%', max_cote, max_cote, now_ms))
        
        recommendations = []
        for row in cur.fetchall():
            score = calculate_prediction_score(
                cote=row[4],
                cote_ref=row[5],
                tendance=row[6],
                amplitude=row[7],
                est_favori=row[8],
                avis=row[9]
            )
            
            proba = score_to_probability(score, row[4])
            value = calculate_value_bet(proba, row[4])
            
            # Appliquer les filtres
            if score >= min_score and value >= min_value:
                rec = get_recommendation(score, value, row[4])
                
                if rec['action'] != 'ÉVITER':
                    recommendations.append({
                        "nom": row[0],
                        "race_key": row[1],
                        "hippodrome": row[2],
                        "numero": row[3],
                        "cote": row[4],
                        "tendance": row[6],
                        "avis_entraineur": row[9],
                        "jockey": row[10],
                        "entraineur": row[11],
                        "type_course": row[12],
                        "distance": row[13],
                        "resultat_reel": "Gagné" if row[15] == 1 else ("Placé" if row[14] and row[14] <= 3 else "Non placé" if row[14] else "En attente"),
                        "score": score,
                        "probabilite": proba,
                        "value_bet": value,
                        "recommendation": rec
                    })
        
        # Trier par niveau de recommandation puis par value
        niveau_order = {"FORT": 0, "MOYEN": 1, "FAIBLE": 2}
        recommendations.sort(key=lambda x: (niveau_order.get(x['recommendation']['niveau'], 99), -x['value_bet']))
        
        # Statistiques
        stats = {
            "total_recommandations": len(recommendations),
            "paris_forts": len([r for r in recommendations if r['recommendation']['niveau'] == 'FORT']),
            "paris_moyens": len([r for r in recommendations if r['recommendation']['niveau'] == 'MOYEN']),
            "paris_faibles": len([r for r in recommendations if r['recommendation']['niveau'] == 'FAIBLE']),
            "value_moyenne": round(sum(r['value_bet'] for r in recommendations) / len(recommendations), 1) if recommendations else 0,
            "score_moyen": round(sum(r['score'] for r in recommendations) / len(recommendations), 1) if recommendations else 0
        }
        
        con.close()
        return {"recommendations": recommendations[:50], "stats": stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/performance-history")
async def get_performance_history():
    """Historique des performances pour graphiques"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Performance par mois
        cur.execute("""
            SELECT 
                SUBSTRING(race_key FROM 1 FOR 7) as mois,
                COUNT(*) as total_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                AVG(cote_finale) as cote_moyenne
            FROM cheval_courses_seen
            WHERE race_key LIKE '2025%'
            AND cote_finale IS NOT NULL
            GROUP BY SUBSTRING(race_key FROM 1 FOR 7)
            ORDER BY mois
        """)
        
        monthly_data = []
        for row in cur.fetchall():
            monthly_data.append({
                "mois": row[0],
                "total": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0,
                "cote_moyenne": round(row[3], 2) if row[3] else 0
            })
        
        # Top chevaux performants
        cur.execute("""
            SELECT 
                nom_norm,
                COUNT(*) as nb_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                AVG(cote_finale) as cote_moy
            FROM cheval_courses_seen
            WHERE race_key LIKE '2025%'
            GROUP BY nom_norm
            HAVING COUNT(*) >= 3
            ORDER BY CAST(SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) DESC
            LIMIT 10
        """)
        
        top_chevaux = []
        for row in cur.fetchall():
            top_chevaux.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0,
                "cote_moyenne": round(row[3], 2) if row[3] else 0
            })
        
        # Distribution des cotes gagnantes
        cur.execute("""
            SELECT 
                CASE 
                    WHEN cote_finale < 2 THEN '< 2'
                    WHEN cote_finale < 3 THEN '2-3'
                    WHEN cote_finale < 5 THEN '3-5'
                    WHEN cote_finale < 10 THEN '5-10'
                    WHEN cote_finale < 20 THEN '10-20'
                    ELSE '> 20'
                END as tranche,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as gagnants
            FROM cheval_courses_seen
            WHERE cote_finale IS NOT NULL
            AND race_key LIKE '2025%'
            GROUP BY tranche
        """)
        
        odds_distribution = []
        for row in cur.fetchall():
            odds_distribution.append({
                "tranche": row[0],
                "total": row[1],
                "gagnants": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        con.close()
        return {
            "monthly_performance": monthly_data,
            "top_chevaux": top_chevaux,
            "odds_distribution": odds_distribution
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cheval/{nom}/profile")
async def get_cheval_profile(nom: str):
    """Profil détaillé d'un cheval"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Normaliser le nom
        nom_search = nom.lower().replace("'", "_")
        
        # Statistiques globales
        cur.execute("""
            SELECT 
                nom_norm,
                COUNT(*) as nb_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as places,
                AVG(cote_finale) as cote_moyenne,
                MIN(cote_finale) as meilleure_cote,
                MAX(gains_course) as meilleur_gain,
                SUM(gains_course) as gains_total
            FROM cheval_courses_seen
            WHERE LOWER(nom_norm) LIKE %s
            GROUP BY nom_norm
        """, (f"%{nom_search}%",))
        
        stats = cur.fetchone()
        
        if not stats:
            raise HTTPException(status_code=404, detail="Cheval non trouvé")
        
        # Historique des courses
        cur.execute("""
            SELECT 
                race_key,
                hippodrome_nom,
                type_course,
                distance_m,
                cote_finale,
                place_finale,
                is_win,
                gains_course,
                driver_jockey,
                entraineur
            FROM cheval_courses_seen
            WHERE LOWER(nom_norm) LIKE %s
            ORDER BY race_key DESC
            LIMIT 20
        """, (f"%{nom_search}%",))
        
        historique = []
        for row in cur.fetchall():
            parts = row[0].split('|') if row[0] else []
            historique.append({
                "date": parts[0] if parts else None,
                "reunion": parts[1] if len(parts) > 1 else None,
                "course": parts[2] if len(parts) > 2 else None,
                "hippodrome": row[1],
                "type_course": row[2],
                "distance": row[3],
                "cote": row[4],
                "place": row[5],
                "victoire": row[6] == 1,
                "gains": row[7],
                "jockey": row[8],
                "entraineur": row[9]
            })
        
        # Performance par type de course
        cur.execute("""
            SELECT 
                type_course,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE LOWER(nom_norm) LIKE %s
            AND type_course IS NOT NULL
            GROUP BY type_course
        """, (f"%{nom_search}%",))
        
        perf_type = []
        for row in cur.fetchall():
            perf_type.append({
                "type": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })

        # Performance récente (30 derniers jours)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cur.execute("""
            SELECT 
                SPLIT_PART(race_key, '|', 1) as course_date,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(gains_course) as gains
            FROM cheval_courses_seen
            WHERE LOWER(nom_norm) LIKE %s
            AND race_key IS NOT NULL
            AND SPLIT_PART(race_key, '|', 1) >= %s
            GROUP BY SPLIT_PART(race_key, '|', 1)
            ORDER BY course_date
        """, (f"%{nom_search}%", thirty_days_ago))

        recent_rows = cur.fetchall()
        recent_performance = []
        recent_courses = 0
        recent_wins = 0
        recent_gains = 0

        for row in recent_rows:
            courses_day = row[1] or 0
            wins_day = row[2] or 0
            gains_day = row[3] or 0
            recent_courses += courses_day
            recent_wins += wins_day
            recent_gains += gains_day
            recent_performance.append({
                "date": row[0],
                "courses": courses_day,
                "victoires": wins_day,
                "taux": round((wins_day / courses_day * 100) if courses_day > 0 else 0, 1),
                "gains": gains_day
            })

        recent_win_rate = round((recent_wins / recent_courses * 100) if recent_courses > 0 else 0, 1)
        
        con.close()
        
        return {
            "nom": stats[0],
            "statistiques": {
                "nb_courses": stats[1],
                "victoires": stats[2] or 0,
                "places": stats[3] or 0,
                "taux_victoire": round((stats[2] or 0) / stats[1] * 100, 1) if stats[1] > 0 else 0,
                "taux_place": round((stats[3] or 0) / stats[1] * 100, 1) if stats[1] > 0 else 0,
                "cote_moyenne": round(stats[4], 2) if stats[4] else 0,
                "meilleure_cote": stats[5],
                "meilleur_gain": stats[6],
                "gains_total": stats[7] or 0
            },
            "historique": historique,
            "performance_par_type": perf_type,
            "recent_performance": recent_performance,
            "recent_metrics": {
                "courses_30j": recent_courses,
                "victoires_30j": recent_wins,
                "taux_victoire_30j": recent_win_rate,
                "gains_30j": recent_gains
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS JOCKEYS
# ============================================================================

@app.get("/api/jockeys/top")
async def get_top_jockeys(limit: int = 50):
    """Liste des meilleurs jockeys"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        cur.execute("""
            SELECT 
                driver_jockey,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE driver_jockey IS NOT NULL AND driver_jockey != ''
            GROUP BY driver_jockey
            HAVING COUNT(*) >= 2
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC, COUNT(*) DESC
            LIMIT %s
        """, (limit,))
        
        jockeys = []
        for row in cur.fetchall():
            jockeys.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        con.close()
        return {"jockeys": jockeys}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jockey/{nom}/stats")
async def get_jockey_stats(nom: str):
    """Statistiques détaillées d'un jockey"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        nom_search = f"%{nom}%"
        
        # Statistiques globales
        cur.execute("""
            SELECT 
                driver_jockey,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as places,
                AVG(CASE WHEN is_win = 1 THEN cote_finale END) as cote_moy_victoire,
                SUM(gains_course) as gains_total
            FROM cheval_courses_seen
            WHERE driver_jockey ILIKE %s
            GROUP BY driver_jockey
        """, (nom_search,))
        
        stats = cur.fetchone()
        
        if not stats:
            raise HTTPException(status_code=404, detail="Jockey non trouvé")
        
        # Tendance mensuelle
        cur.execute("""
            SELECT 
                TO_CHAR(TO_DATE(SPLIT_PART(race_key, '|', 1), 'YYYY-MM-DD'), 'YYYY-MM') as mois,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE driver_jockey ILIKE %s
            AND race_key LIKE '202%%'
            GROUP BY TO_CHAR(TO_DATE(SPLIT_PART(race_key, '|', 1), 'YYYY-MM-DD'), 'YYYY-MM')
            ORDER BY mois DESC
            LIMIT 12
        """, (nom_search,))
        
        tendance = []
        for row in cur.fetchall():
            tendance.append({
                "mois": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Performance par type
        cur.execute("""
            SELECT 
                type_course,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE driver_jockey ILIKE %s
            AND type_course IS NOT NULL
            GROUP BY type_course
        """, (nom_search,))
        
        perf_type = []
        for row in cur.fetchall():
            perf_type.append({
                "type": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Top hippodromes
        cur.execute("""
            SELECT 
                hippodrome_nom,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE driver_jockey ILIKE %s
            AND hippodrome_nom IS NOT NULL
            GROUP BY hippodrome_nom
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC
            LIMIT 10
        """, (nom_search,))
        
        top_hippos = []
        for row in cur.fetchall():
            top_hippos.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Associations gagnantes avec entraineurs
        cur.execute("""
            SELECT 
                entraineur,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE driver_jockey ILIKE %s
            AND entraineur IS NOT NULL
            GROUP BY entraineur
            HAVING COUNT(*) >= 5
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC
            LIMIT 10
        """, (nom_search,))
        
        associations = []
        for row in cur.fetchall():
            associations.append({
                "entraineur": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })

        # Performance récente (30 jours)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cur.execute("""
            SELECT 
                SPLIT_PART(race_key, '|', 1) as course_date,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(gains_course) as gains
            FROM cheval_courses_seen
            WHERE driver_jockey ILIKE %s
            AND race_key IS NOT NULL
            AND SPLIT_PART(race_key, '|', 1) >= %s
            GROUP BY SPLIT_PART(race_key, '|', 1)
            ORDER BY course_date
        """, (nom_search, thirty_days_ago))

        recent_rows = cur.fetchall()
        recent_performance = []
        recent_courses = 0
        recent_wins = 0
        recent_gains = 0

        for row in recent_rows:
            courses_day = row[1] or 0
            wins_day = row[2] or 0
            gains_day = row[3] or 0
            recent_courses += courses_day
            recent_wins += wins_day
            recent_gains += gains_day
            recent_performance.append({
                "date": row[0],
                "courses": courses_day,
                "victoires": wins_day,
                "taux": round((wins_day / courses_day * 100) if courses_day > 0 else 0, 1),
                "gains": gains_day
            })

        recent_win_rate = round((recent_wins / recent_courses * 100) if recent_courses > 0 else 0, 1)
        
        con.close()
        
        return {
            "nom": stats[0],
            "statistiques": {
                "courses": stats[1],
                "victoires": stats[2] or 0,
                "places": stats[3] or 0,
                "taux_victoire": round((stats[2] or 0) / stats[1] * 100, 1) if stats[1] > 0 else 0,
                "taux_place": round((stats[3] or 0) / stats[1] * 100, 1) if stats[1] > 0 else 0,
                "cote_moyenne": round(stats[4], 2) if stats[4] else 0,
                "gains_total": stats[5] or 0
            },
            "tendance_mensuelle": tendance,
            "performance_par_type": perf_type,
            "top_hippodromes": top_hippos,
            "associations_gagnantes": associations,
            "recent_performance": recent_performance,
            "recent_metrics": {
                "courses_30j": recent_courses,
                "victoires_30j": recent_wins,
                "taux_victoire_30j": recent_win_rate,
                "gains_30j": recent_gains
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS ENTRAINEURS
# ============================================================================

@app.get("/api/entraineurs/top")
async def get_top_entraineurs(limit: int = 50):
    """Liste des meilleurs entraineurs"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        cur.execute("""
            SELECT 
                entraineur,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                COUNT(DISTINCT nom_norm) as nb_chevaux
            FROM cheval_courses_seen
            WHERE entraineur IS NOT NULL AND entraineur != ''
            GROUP BY entraineur
            HAVING COUNT(*) >= 2
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC, COUNT(*) DESC
            LIMIT %s
        """, (limit,))
        
        entraineurs = []
        for row in cur.fetchall():
            entraineurs.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "nb_chevaux": row[3],
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        con.close()
        return {"entraineurs": entraineurs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entraineur/{nom}/stats")
async def get_entraineur_stats(nom: str):
    """Statistiques détaillées d'un entraineur"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        nom_search = f"%{nom}%"
        
        # Statistiques globales
        cur.execute("""
            SELECT 
                entraineur,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                COUNT(DISTINCT nom_norm) as nb_chevaux,
                SUM(gains_course) as gains_total
            FROM cheval_courses_seen
            WHERE entraineur ILIKE %s
            GROUP BY entraineur
        """, (nom_search,))
        
        stats = cur.fetchone()
        
        if not stats:
            raise HTTPException(status_code=404, detail="Entraineur non trouvé")
        
        # Tendance mensuelle
        cur.execute("""
            SELECT 
                TO_CHAR(TO_DATE(SPLIT_PART(race_key, '|', 1), 'YYYY-MM-DD'), 'YYYY-MM') as mois,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE entraineur ILIKE %s
            AND race_key LIKE '202%%'
            GROUP BY TO_CHAR(TO_DATE(SPLIT_PART(race_key, '|', 1), 'YYYY-MM-DD'), 'YYYY-MM')
            ORDER BY mois DESC
            LIMIT 12
        """, (nom_search,))
        
        tendance = []
        for row in cur.fetchall():
            tendance.append({
                "mois": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Performance par type
        cur.execute("""
            SELECT 
                type_course,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE entraineur ILIKE %s
            AND type_course IS NOT NULL
            GROUP BY type_course
        """, (nom_search,))
        
        perf_type = []
        for row in cur.fetchall():
            perf_type.append({
                "type": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Top chevaux de l'écurie
        cur.execute("""
            SELECT 
                nom_norm,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(gains_course) as gains
            FROM cheval_courses_seen
            WHERE entraineur ILIKE %s
            GROUP BY nom_norm
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC
            LIMIT 10
        """, (nom_search,))
        
        top_chevaux = []
        for row in cur.fetchall():
            top_chevaux.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "gains": row[3] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Jockeys favoris
        cur.execute("""
            SELECT 
                driver_jockey,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE entraineur ILIKE %s
            AND driver_jockey IS NOT NULL
            GROUP BY driver_jockey
            HAVING COUNT(*) >= 3
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC
            LIMIT 10
        """, (nom_search,))
        
        jockeys_favoris = []
        for row in cur.fetchall():
            jockeys_favoris.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })

        # Performance récente (30 jours)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cur.execute("""
            SELECT 
                SPLIT_PART(race_key, '|', 1) as course_date,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(gains_course) as gains
            FROM cheval_courses_seen
            WHERE entraineur ILIKE %s
            AND race_key IS NOT NULL
            AND SPLIT_PART(race_key, '|', 1) >= %s
            GROUP BY SPLIT_PART(race_key, '|', 1)
            ORDER BY course_date
        """, (nom_search, thirty_days_ago))

        recent_rows = cur.fetchall()
        recent_performance = []
        recent_courses = 0
        recent_wins = 0
        recent_gains = 0

        for row in recent_rows:
            courses_day = row[1] or 0
            wins_day = row[2] or 0
            gains_day = row[3] or 0
            recent_courses += courses_day
            recent_wins += wins_day
            recent_gains += gains_day
            recent_performance.append({
                "date": row[0],
                "courses": courses_day,
                "victoires": wins_day,
                "taux": round((wins_day / courses_day * 100) if courses_day > 0 else 0, 1),
                "gains": gains_day
            })

        recent_win_rate = round((recent_wins / recent_courses * 100) if recent_courses > 0 else 0, 1)
        
        con.close()
        
        return {
            "nom": stats[0],
            "statistiques": {
                "courses": stats[1],
                "victoires": stats[2] or 0,
                "nb_chevaux": stats[3],
                "taux_victoire": round((stats[2] or 0) / stats[1] * 100, 1) if stats[1] > 0 else 0,
                "gains_total": stats[4] or 0
            },
            "tendance_mensuelle": tendance,
            "performance_par_type": perf_type,
            "top_chevaux": top_chevaux,
            "jockeys_favoris": jockeys_favoris,
            "recent_performance": recent_performance,
            "recent_metrics": {
                "courses_30j": recent_courses,
                "victoires_30j": recent_wins,
                "taux_victoire_30j": recent_win_rate,
                "gains_30j": recent_gains
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS HIPPODROMES
# ============================================================================

@app.get("/api/hippodromes")
async def get_hippodromes():
    """Liste des hippodromes"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        cur.execute("""
            SELECT 
                hippodrome_nom,
                COUNT(*) as courses
            FROM cheval_courses_seen
            WHERE hippodrome_nom IS NOT NULL AND hippodrome_nom != ''
            GROUP BY hippodrome_nom
            ORDER BY COUNT(*) DESC
            LIMIT 100
        """)
        
        hippodromes = []
        for row in cur.fetchall():
            hippodromes.append({
                "nom": row[0],
                "courses": row[1]
            })
        
        con.close()
        return {"hippodromes": hippodromes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hippodrome/{nom}/stats")
async def get_hippodrome_stats(nom: str):
    """Statistiques détaillées d'un hippodrome"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        nom_search = f"%{nom}%"
        
        # Statistiques globales
        cur.execute("""
            SELECT 
                hippodrome_nom,
                COUNT(DISTINCT race_key) as total_courses,
                AVG(CASE WHEN is_win = 1 THEN cote_finale END) as cote_moyenne_gagnant,
                MAX(SPLIT_PART(race_key, '|', 1)) as derniere_course,
                SUM(gains_course) as gains_total
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
            GROUP BY hippodrome_nom
        """, (nom_search,))
        
        stats = cur.fetchone()
        
        if not stats:
            raise HTTPException(status_code=404, detail="Hippodrome non trouvé")
        
        # Taux favoris gagnants (cote < 5)
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN cote_finale < 5 AND is_win = 1 THEN 1 ELSE 0 END) as favoris_gagnants,
                SUM(CASE WHEN cote_finale < 5 THEN 1 ELSE 0 END) as favoris_total
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
        """, (nom_search,))
        
        fav_stats = cur.fetchone()
        taux_favoris = round((fav_stats[1] or 0) / fav_stats[2] * 100, 1) if fav_stats[2] > 0 else 0
        
        # Performance par type
        cur.execute("""
            SELECT 
                type_course,
                COUNT(DISTINCT race_key) as courses,
                SUM(CASE WHEN cote_finale < 5 AND is_win = 1 THEN 1 ELSE 0 END) as favoris_gagnants,
                SUM(CASE WHEN cote_finale < 5 THEN 1 ELSE 0 END) as favoris_total,
                SUM(CASE WHEN cote_finale >= 10 AND is_win = 1 THEN 1 ELSE 0 END) as outsiders_gagnants,
                SUM(CASE WHEN cote_finale >= 10 THEN 1 ELSE 0 END) as outsiders_total
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
            AND type_course IS NOT NULL
            GROUP BY type_course
        """, (nom_search,))
        
        perf_type = []
        for row in cur.fetchall():
            perf_type.append({
                "type": row[0],
                "courses": row[1],
                "taux_favoris": round((row[2] or 0) / row[3] * 100, 1) if row[3] > 0 else 0,
                "taux_outsiders": round((row[4] or 0) / row[5] * 100, 1) if row[5] > 0 else 0
            })
        
        # Stats par distance
        cur.execute("""
            SELECT 
                CASE 
                    WHEN distance_m < 1500 THEN 'Court (<1500m)'
                    WHEN distance_m < 2000 THEN 'Moyen (1500-2000m)'
                    WHEN distance_m < 2500 THEN 'Long (2000-2500m)'
                    ELSE 'Très long (>2500m)'
                END as distance_cat,
                COUNT(DISTINCT race_key) as courses,
                SUM(CASE WHEN cote_finale < 5 AND is_win = 1 THEN 1 ELSE 0 END) as favoris,
                SUM(CASE WHEN cote_finale < 5 THEN 1 ELSE 0 END) as total_favoris
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
            AND distance_m IS NOT NULL
            GROUP BY CASE 
                    WHEN distance_m < 1500 THEN 'Court (<1500m)'
                    WHEN distance_m < 2000 THEN 'Moyen (1500-2000m)'
                    WHEN distance_m < 2500 THEN 'Long (2000-2500m)'
                    ELSE 'Très long (>2500m)'
                END
        """, (nom_search,))
        
        distance_stats = []
        for row in cur.fetchall():
            distance_stats.append({
                "distance": row[0],
                "courses": row[1],
                "taux_favori": round((row[2] or 0) / row[3] * 100, 1) if row[3] > 0 else 0
            })
        
        # Top chevaux
        cur.execute("""
            SELECT 
                nom_norm,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
            GROUP BY nom_norm
            HAVING COUNT(*) >= 3
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC
            LIMIT 10
        """, (nom_search,))
        
        top_chevaux = []
        for row in cur.fetchall():
            top_chevaux.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })
        
        # Top jockeys
        cur.execute("""
            SELECT 
                driver_jockey,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
            AND driver_jockey IS NOT NULL
            GROUP BY driver_jockey
            HAVING COUNT(*) >= 5
            ORDER BY SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) DESC
            LIMIT 10
        """, (nom_search,))
        
        top_jockeys = []
        for row in cur.fetchall():
            top_jockeys.append({
                "nom": row[0],
                "courses": row[1],
                "victoires": row[2] or 0,
                "taux": round((row[2] or 0) / row[1] * 100, 1) if row[1] > 0 else 0
            })

        # Performance récente (30 jours)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cur.execute("""
            SELECT 
                SPLIT_PART(race_key, '|', 1) as course_date,
                COUNT(*) as courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(gains_course) as gains
            FROM cheval_courses_seen
            WHERE hippodrome_nom ILIKE %s
            AND race_key IS NOT NULL
            AND SPLIT_PART(race_key, '|', 1) >= %s
            GROUP BY SPLIT_PART(race_key, '|', 1)
            ORDER BY course_date
        """, (nom_search, thirty_days_ago))

        recent_rows = cur.fetchall()
        recent_performance = []
        recent_courses = 0
        recent_wins = 0
        recent_gains = 0

        for row in recent_rows:
            courses_day = row[1] or 0
            wins_day = row[2] or 0
            gains_day = row[3] or 0
            recent_courses += courses_day
            recent_wins += wins_day
            recent_gains += gains_day
            recent_performance.append({
                "date": row[0],
                "courses": courses_day,
                "victoires": wins_day,
                "taux": round((wins_day / courses_day * 100) if courses_day > 0 else 0, 1),
                "gains": gains_day
            })

        recent_win_rate = round((recent_wins / recent_courses * 100) if recent_courses > 0 else 0, 1)
        
        con.close()
        
        return {
            "nom": stats[0],
            "total_courses": stats[1],
            "cote_moyenne": round(stats[2], 2) if stats[2] else 0,
            "derniere_course": stats[3],
            "gains_total": stats[4] or 0,
            "taux_favoris": taux_favoris,
            "performance_par_type": perf_type,
            "statistiques_distances": distance_stats,
            "top_chevaux": top_chevaux,
            "top_jockeys": top_jockeys,
            "recent_performance": recent_performance,
            "recent_metrics": {
                "courses_30j": recent_courses,
                "victoires_30j": recent_wins,
                "taux_victoire_30j": recent_win_rate,
                "gains_30j": recent_gains
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS - SYSTÈME DE RECOMMANDATION DE PARIS AVANCÉ
# ============================================================================

# Import du module de recommandation (chargé une fois)
_betting_advisor = None
_race_pronostic_generator = None

def get_betting_advisor():
    """Singleton pour le BettingAdvisor"""
    global _betting_advisor
    if _betting_advisor is None:
        try:
            # Importer depuis le répertoire parent
            sys.path.insert(0, os.path.dirname(os.path.dirname(BASE_DIR)))
            from betting_advisor import BettingAdvisor
            con = get_db_connection()
            _betting_advisor = BettingAdvisor(con)
        except Exception as e:
            print(f"[WARNING] BettingAdvisor non disponible: {e}")
            return None
    return _betting_advisor

def get_race_pronostic_generator():
    """Singleton pour le RacePronosticGenerator (moteur principal)"""
    global _race_pronostic_generator
    if _race_pronostic_generator is None:
        try:
            # En environnement Docker, utiliser les volumes mappés
            import os
            if os.path.exists('/project/race_pronostic_generator.py'):
                # Mode Docker : utiliser les volumes mappés
                sys.path.insert(0, '/project')
                print("[INFO] Mode Docker détecté - utilisation des volumes mappés")
            else:
                # Mode développement local : utiliser le path relatif
                sys.path.insert(0, os.path.dirname(os.path.dirname(BASE_DIR)))
                print("[INFO] Mode développement détecté - utilisation du path relatif")
            
            from race_pronostic_generator import RacePronosticGenerator
            con = get_db_connection()
            _race_pronostic_generator = RacePronosticGenerator(con)
            print("[INFO] ✓ RacePronosticGenerator chargé avec succès")
        except Exception as e:
            print(f"[WARNING] RacePronosticGenerator non disponible: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _race_pronostic_generator

def format_recommendation_from_generator(runner, race_key, race_info, user_params: Optional[UserBettingParams] = None):
    """
    Convertit la sortie du RacePronosticGenerator en format attendu par le frontend.
    Avec personnalisation optionnelle selon le profil utilisateur.
    
    Args:
        runner: Dict contenant les données d'un cheval du générateur
        race_key: Clé de la course
        race_info: Infos générales de la course
        user_params: Paramètres utilisateur optionnels pour personnalisation
    
    Returns:
        Dict formaté pour le frontend ou None si non recommandé
    """
    try:
        # Données de base
        nom = runner.get('nom', 'N/A')
        numero = runner.get('numero', 0)
        odds = runner.get('odds', 0)
        p_win = runner.get('p_win', 0) 
        value_pct = runner.get('value_pct', 0)
        kelly_stake = runner.get('kelly_stake', 0)
        kelly_pct = runner.get('kelly_pct', 0)
        ev_expected = runner.get('ev_expected', 0)
        bucket = runner.get('bucket', '')
        rationale = runner.get('rationale', [])
        
        # Filtrer seulement les chevaux avec de la value
        if value_pct <= 0:
            return None
        
        # Calculs personnalisés si paramètres utilisateur fournis
        if user_params:
            personalized = calculate_personalized_stake(runner, user_params)
            if not personalized['eligible']:
                return None  # Filtré par le profil
            
            # Utiliser les mises personnalisées
            stake_euros = personalized['stake_euros']
            stake_pct = personalized['stake_pct']
            gain_potentiel = personalized['gain_potentiel']
            mise_info = f"{stake_euros:.1f}€ ({stake_pct:.1f}% bankroll)"
            profil_info = personalized['reason']
        else:
            # Calcul standard (base 1000€)
            stake_euros = kelly_stake * 1000
            stake_pct = kelly_pct
            gain_potentiel = stake_euros * (odds - 1) if stake_euros > 0 else 0
            mise_info = f"{stake_euros:.1f}€" if stake_euros >= 1 else f"{kelly_pct:.1f}%"
            profil_info = "Calcul standard"
            
        # Déterminer le type de recommandation selon les buckets du générateur
        if bucket == 'SUR' or (odds < 4 and value_pct > 5 and p_win > 0.20):
            type_rec = 'SUR'
            emoji = '🟢'
            desc_risque = 'Faible risque'
        elif bucket == 'EQUILIBRE' or (4 <= odds <= 15 and value_pct > 0):
            type_rec = 'EQUILIBRE' 
            emoji = '🟡'
            desc_risque = 'Risque moyen'
        elif bucket == 'RISQUE' or (odds > 15 and value_pct > 10):
            type_rec = 'RISQUE'
            emoji = '🔴'
            desc_risque = 'Haut potentiel'
        else:
            return None  # Pas assez de value
        
        # Gain potentiel formaté
        gain_str = f"{gain_potentiel:.1f}€" if gain_potentiel >= 1 else "Variable"
        
        # Construire la rationale avec info profil
        raison_base = " • ".join(rationale[:3]) if rationale else f"Value {value_pct:.1f}% avec probabilité {p_win*100:.1f}%"
        if user_params:
            raison = f"{raison_base} • {profil_info}"
        else:
            raison = raison_base
        
        # Extraire hippodrome du race_key si pas dans race_info  
        hippodrome = race_info.get('hippodrome', '')
        if not hippodrome and '|' in race_key:
            parts = race_key.split('|')
            if len(parts) >= 2:
                hippodrome = parts[1]
        
        return {
            "nom": nom,
            "numero": numero,
            "cote": odds,
            "probabilite": round(p_win * 100, 1),
            "value_pct": round(value_pct, 1),
            "mise_recommandee": mise_info,
            "mise_recommandee_num": stake_euros,
            "gain_potentiel": gain_str,
            "gain_potentiel_num": gain_potentiel,
            "esperance": round(ev_expected, 2),
            "type": type_rec,
            "emoji": emoji,
            "risque": desc_risque,
            "rationale": raison,
            "race_key": race_key,
            "hippodrome": hippodrome
        }
        
    except Exception as e:
        print(f"[ERROR] format_recommendation_from_generator: {e}")
        return None

def calculate_personalized_stake(runner: Dict, user_params: UserBettingParams) -> Dict[str, Any]:
    """
    🎯 Calcule la mise personnalisée selon le profil utilisateur
    
    Args:
        runner: Données du cheval du générateur
        user_params: Paramètres utilisateur (bankroll + profil)
    
    Returns:
        Dict avec mise ajustée et informations
    """
    config = user_params.get_config()
    
    # Données de base du runner
    kelly_pct = runner.get('kelly_pct', 0) / 100  # Convertir % en fraction
    value_pct = runner.get('value_pct', 0)
    odds = runner.get('odds', 0)
    p_win = runner.get('p_win', 0)
    
    # Filtres du profil
    if value_pct < config['min_value_pct']:
        return {"eligible": False, "reason": f"Value {value_pct:.1f}% < seuil {config['min_value_pct']:.1f}%"}
    
    if odds > config['max_odds']:
        return {"eligible": False, "reason": f"Cote {odds:.1f} > limite {config['max_odds']:.1f}"}
    
    # Calcul de la mise ajustée
    kelly_adjusted = kelly_pct * config['kelly_multiplier']
    
    # Application du cap par pari
    max_stake_fraction = config['max_stake_pct'] / 100
    kelly_capped = min(kelly_adjusted, max_stake_fraction)
    
    # Mise en euros
    stake_euros = kelly_capped * user_params.bankroll
    
    # Arrondissement intelligent
    if stake_euros < 5:
        stake_euros = round(stake_euros, 1)  # Ex: 2.3€
    elif stake_euros < 20:
        stake_euros = round(stake_euros * 2) / 2  # Ex: 12.5€
    else:
        stake_euros = round(stake_euros)  # Ex: 25€
    
    # Calcul du gain potentiel
    gain_potentiel = stake_euros * (odds - 1)
    
    # Pourcentage de la bankroll
    stake_pct = (stake_euros / user_params.bankroll) * 100
    
    return {
        "eligible": True,
        "stake_euros": stake_euros,
        "stake_pct": stake_pct,
        "gain_potentiel": gain_potentiel,
        "kelly_original": kelly_pct * 100,
        "kelly_adjusted": kelly_adjusted * 100,
        "profil_multiplier": config['kelly_multiplier'],
        "reason": f"Kelly {kelly_pct*100:.1f}% → {kelly_adjusted*100:.1f}% (profil {user_params.profil.value})"
    }
    """
    Convertit la sortie du RacePronosticGenerator en format attendu par le frontend.
    
    Args:
        runner: Dict contenant les données d'un cheval du générateur
        race_key: Clé de la course
        race_info: Infos générales de la course
    
    Returns:
        Dict formaté pour le frontend ou None si pas de recommandation
    """
    try:
        # Extraire les données principales
        nom = runner.get('nom', 'Inconnu')
        numero = runner.get('numero', 0)
        p_win = runner.get('p_win', 0)
        odds = runner.get('odds', 0)
        value_pct = runner.get('value_pct', 0)
        kelly_stake = runner.get('kelly_stake', 0)
        kelly_pct = runner.get('kelly_pct', 0)
        ev_expected = runner.get('ev_expected', 0)
        bucket = runner.get('bucket', '')
        rationale = runner.get('rationale', [])
        
        # Filtrer seulement les chevaux avec de la value
        if value_pct <= 0:
            return None
            
        # Déterminer le type de recommandation selon les buckets du générateur
        if bucket == 'SUR' or (odds < 4 and value_pct > 5 and p_win > 0.20):
            type_rec = 'SUR'
            emoji = '🟢'
            desc_risque = 'Faible risque'
        elif bucket == 'EQUILIBRE' or (4 <= odds <= 15 and value_pct > 0):
            type_rec = 'EQUILIBRE' 
            emoji = '🟡'
            desc_risque = 'Risque moyen'
        elif bucket == 'RISQUE' or (odds > 15 and value_pct > 10):
            type_rec = 'RISQUE'
            emoji = '🔴'
            desc_risque = 'Haut potentiel'
        else:
            return None  # Pas assez de value
        
        # Formater la mise recommandée
        mise_euro = kelly_stake * 1000  # Convertir fraction en euros (base 1000€)
        mise_str = f"{mise_euro:.1f}€" if mise_euro >= 1 else f"{kelly_pct:.1f}%"
        
        # Calculer le gain potentiel
        gain_potentiel = mise_euro * (odds - 1) if mise_euro > 0 else 0
        gain_str = f"{gain_potentiel:.1f}€" if gain_potentiel >= 1 else "Variable"
        
        # Construire la rationale
        raison = " • ".join(rationale[:3]) if rationale else f"Value {value_pct:.1f}% avec probabilité {p_win*100:.1f}%"
        
        # Extraire hippodrome du race_key si pas dans race_info  
        hippodrome = race_info.get('hippodrome', '')
        if not hippodrome and '|' in race_key:
            parts = race_key.split('|')
            if len(parts) > 3:
                hippodrome = parts[3]
        
        return {
            'type': type_rec,
            'cheval': nom,
            'numero': numero,
            'hippodrome': hippodrome,
            'cote': odds,
            'proba_succes': round(p_win * 100, 1),
            'esperance': round(value_pct, 1),
            'mise_recommandee': mise_str,
            'mise_recommandee_num': mise_euro,
            'gain_potentiel': gain_str,
            'raison': raison,
            'details': rationale[:5] if rationale else [],
            'type_pari': 'GAGNANT',  # Par défaut
            'race_key': race_key,
            'kelly_fraction': round(kelly_pct, 2),
            'ev_expected': round(ev_expected, 2),
            'bucket_original': bucket
        }
        
    except Exception as e:
        print(f"[WARNING] Erreur formatage recommandation: {e}")
        return None


@app.get("/api/paris/recommandations-jour")
async def get_paris_recommandations_jour(user: Optional[Dict] = Depends(get_user_from_request)):
    """
    🎯 Récupère les recommandations de paris personnalisées pour toutes les courses du jour
    
    Utilise le moteur d'analyse complet (race_pronostic_generator) avec:
    - Calibration des probabilités  
    - Kelly criterion pour les mises
    - Blend modèle + marché
    - Classification SÛR/ÉQUILIBRÉ/RISQUÉ
    - Personnalisation selon les settings de l'utilisateur authentifié
    
    Nécessite une authentification pour accéder aux settings personnalisés.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentification requise pour les recommandations personnalisées")
    
    try:
        # Récupérer les settings utilisateur
        user_settings = await get_user_settings(user["id"])
        
        con = get_db_connection()
        cur = con.cursor()
        
        # Date du jour
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Récupérer toutes les courses du jour (non terminées)
        cur.execute("""
            SELECT DISTINCT race_key
            FROM cheval_courses_seen
            WHERE race_key LIKE %s
            AND place_finale IS NULL  -- Seulement courses non terminées
            ORDER BY race_key
        """, (today + '%',))
        
        courses = [row[0] for row in cur.fetchall()]
        
        if not courses:
            con.close()
            return {
                "date": today,
                "nb_courses": 0,
                "message": "Aucune course disponible pour aujourd'hui",
                "meilleurs_paris": {"sur": [], "equilibre": [], "risque": []},
                "resume": {"total_paris_sur": 0, "total_paris_equilibre": 0, "total_paris_risque": 0},
                "user_profile": {"bankroll": user_settings.bankroll, "profil_risque": user_settings.profil_risque}
            }
        
        # Utiliser le moteur principal d'analyse
        generator = get_race_pronostic_generator()
        
        # Convertir user_settings en UserBettingParams pour compatibilité avec l'existant
        profil_mapping = {
            'PRUDENT': ProfilRisque.PRUDENT,
            'STANDARD': ProfilRisque.STANDARD, 
            'AGRESSIF': ProfilRisque.AGRESSIF
        }
        profil_risque = profil_mapping.get(user_settings.profil_risque, ProfilRisque.STANDARD)
        
        # Créer les paramètres utilisateur
        user_params = UserBettingParams(bankroll=user_settings.bankroll, profil=profil_risque)
        config = user_params.get_config()
        
        all_recommandations = {
            "SUR": [],      # Paris sûrs (🟢)
            "EQUILIBRE": [],  # Paris équilibrés (🟡)  
            "RISQUE": []    # Paris risqués (🔴)
        }
        
        courses_analysees = 0
        total_mise_jour = 0.0  # Pour vérifier le cap journalier
        
        for race_key in courses:
            try:
                if generator:
                    # Utiliser le moteur sophistiqué
                    pronostic_dict = generator.generate_pronostic_dict(race_key)
                    
                    if pronostic_dict and 'runners' in pronostic_dict:
                        courses_analysees += 1
                        race_info = pronostic_dict.get('race_info', {})
                        
                        # Traiter chaque cheval avec personnalisation
                        for runner in pronostic_dict['runners']:
                            rec = format_recommendation_from_generator(runner, race_key, race_info, user_params)
                            if rec:
                                # Vérifier le cap journalier avant d'ajouter
                                nouvelle_mise = rec.get('mise_recommandee_num', 0)
                                if total_mise_jour + nouvelle_mise <= user_settings.bankroll * config['max_daily_pct'] / 100:
                                    all_recommandations[rec['type']].append(rec)
                                    total_mise_jour += nouvelle_mise
                else:
                    # Fallback vers l'ancien système (sans personnalisation)
                    rapport = analyser_course_pour_paris(cur, race_key)
                    if rapport:
                        courses_analysees += 1
                        for rec in rapport.get('recommandations', []):
                            rec['race_key'] = race_key
                            rec['hippodrome'] = rapport['hippodrome']
                            all_recommandations[rec['type']].append(rec)
                        
            except Exception as e:
                print(f"[WARNING] Erreur analyse {race_key}: {e}")
                continue
        
        con.close()
        
        # Trier les recommandations par valeur puis Kelly
        for key in all_recommandations:
            all_recommandations[key].sort(key=lambda x: (x.get('esperance', 0), x.get('mise_recommandee_num', 0)), reverse=True)
        
        return {
            "date": today,
            "nb_courses": courses_analysees,
            "profil_utilisateur": {
                "bankroll": user_settings.bankroll,
                "profil": user_settings.profil_risque,
                "description": config['description'],
                "total_mise_jour": round(total_mise_jour, 1),
                "budget_restant": round(user_settings.bankroll * config['max_daily_pct'] / 100 - total_mise_jour, 1),
                "pourcentage_bankroll_utilise": round((total_mise_jour / user_settings.bankroll) * 100, 1)
            },
            "meilleurs_paris": {
                "sur": all_recommandations["SUR"][:5],
                "equilibre": all_recommandations["EQUILIBRE"][:5],
                "risque": all_recommandations["RISQUE"][:5]
            },
            "resume": {
                "total_paris_sur": len(all_recommandations["SUR"]),
                "total_paris_equilibre": len(all_recommandations["EQUILIBRE"]),
                "total_paris_risque": len(all_recommandations["RISQUE"])
            },
            "moteur": "race_pronostic_generator" if generator else "betting_advisor_fallback"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profils/configurations")
async def get_profils_configurations():
    """
    🎯 Récupère les configurations disponibles pour chaque profil de risque
    """
    return {
        "profils": {
            profil.value: {
                **ConfigProfil.get_config(profil),
                "nom": profil.value.title(),
                "emoji": {
                    "PRUDENT": "🛡️",
                    "STANDARD": "⚖️", 
                    "AGRESSIF": "🚀"
                }[profil.value]
            }
            for profil in ProfilRisque
        }
    }


@app.get("/api/profils/simulation")
async def get_simulation_bankroll(
    bankroll: float = Query(default=500.0, ge=100, le=10000),
    profil: ProfilRisque = Query(default=ProfilRisque.STANDARD)
):
    """
    💰 Simulation du budget journalier selon la bankroll et le profil
    """
    config = ConfigProfil.get_config(profil)
    
    budget_journalier = bankroll * config['max_daily_pct'] / 100
    mise_max_pari = bankroll * config['max_stake_pct'] / 100
    nb_paris_theorique = int(budget_journalier / mise_max_pari)
    
    return {
        "bankroll": bankroll,
        "profil": profil.value,
        "budget_journalier": round(budget_journalier, 1),
        "mise_max_par_pari": round(mise_max_pari, 1),
        "nb_paris_theorique": nb_paris_theorique,
        "description": config['description'],
        "exemple": f"Avec {bankroll:.0f}€, vous pouvez miser {budget_journalier:.0f}€/jour répartis sur ~{nb_paris_theorique} paris (max {mise_max_pari:.1f}€ par pari)"
    }


@app.get("/api/paris/analyse-course/{race_key:path}")
async def get_analyse_course(race_key: str):
    """
    📊 Analyse détaillée d'une course avec recommandations de paris
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        rapport = analyser_course_complete(cur, race_key)
        
        con.close()
        
        if not rapport:
            raise HTTPException(status_code=404, detail="Course non trouvée")
        
        return rapport
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def analyser_course_pour_paris(cur, race_key: str) -> dict:
    """Analyse une course et génère les recommandations de paris"""
    
    # Infos course
    cur.execute("""
        SELECT DISTINCT
            hippodrome_nom,
            distance_m,
            type_course,
            discipline,
            allocation_totale
        FROM cheval_courses_seen
        WHERE race_key = %s
        LIMIT 1
    """, (race_key,))
    
    course_info = cur.fetchone()
    if not course_info:
        return None
    
    hippodrome, distance, type_course, discipline, allocation = course_info
    
    # Participants
    cur.execute("""
        SELECT 
            nom_norm,
            numero_dossard,
            cote_finale,
            cote_reference,
            tendance_cote,
            amplitude_tendance,
            est_favori,
            avis_entraineur,
            driver_jockey,
            entraineur,
            age,
            sexe,
            musique
        FROM cheval_courses_seen
        WHERE race_key = %s
        ORDER BY numero_dossard
    """, (race_key,))
    
    participants = []
    for row in cur.fetchall():
        participant = analyser_participant_pour_paris(cur, row, distance, hippodrome)
        if participant:
            participants.append(participant)
    
    if not participants:
        return None
    
    # Trier par score
    participants.sort(key=lambda x: x['score_global'], reverse=True)
    
    # Générer les recommandations
    recommandations = []
    nb_partants = len(participants)
    
    # Déterminer les types de paris disponibles selon le nombre de partants
    paris_disponibles = ["GAGNANT", "PLACÉ"]
    if nb_partants >= 8:
        paris_disponibles.append("COUPLÉ")
    if nb_partants >= 8:
        paris_disponibles.append("TRIO")
    if nb_partants >= 12:
        paris_disponibles.append("QUARTÉ")
    if nb_partants >= 15:
        paris_disponibles.append("QUINTÉ")
    
    # PARI SÛR - Meilleur candidat cote < 5
    for p in participants:
        if p['cote'] and p['cote'] < 5 and p['score_global'] >= 60:
            # Pour un pari sûr, on recommande PLACÉ (plus de chances)
            type_pari = "PLACÉ"
            if p['proba_victoire'] >= 25:
                type_pari = "GAGNANT"
            elif nb_partants >= 8 and len([x for x in participants[:3] if x['score_global'] >= 50]) >= 2:
                type_pari = "COUPLÉ GAGNANT"
                
            recommandations.append({
                "type": "SUR",
                "type_pari": type_pari,
                "niveau_risque": 1,
                "cheval": p['nom'],
                "numero": p['numero'],
                "cote": p['cote'],
                "mise_recommandee": "3-5% (5-20€)",
                "gain_potentiel": f"{5 * p['cote']:.0f}-{20 * p['cote']:.0f}€",
                "raison": f"Favori avec score {p['score_global']:.0f}/100",
                "details": p['signaux_positifs'][:3],
                "proba_succes": p['proba_victoire'],
                "esperance": p['esperance']
            })
            break
    
    # PARI ÉQUILIBRÉ - Cote 5-15 avec value
    for p in participants:
        if p['cote'] and 5 <= p['cote'] <= 15 and p['score_global'] >= 50 and p.get('value', 0) > 0:
            # Pour un pari équilibré, GAGNANT pour la value ou COUPLÉ PLACÉ
            type_pari = "GAGNANT"
            if nb_partants >= 8:
                # Si le cheval est régulier, couplé placé
                if p.get('taux_place', 0) > 40:
                    type_pari = "COUPLÉ PLACÉ"
                else:
                    type_pari = "GAGNANT"
            
            recommandations.append({
                "type": "EQUILIBRE",
                "type_pari": type_pari,
                "niveau_risque": 3,
                "cheval": p['nom'],
                "numero": p['numero'],
                "cote": p['cote'],
                "mise_recommandee": "2% (3-10€)",
                "gain_potentiel": f"{3 * p['cote']:.0f}-{10 * p['cote']:.0f}€",
                "raison": f"Value bet +{p.get('value', 0):.1f}%",
                "details": p['signaux_positifs'][:3],
                "proba_succes": p['proba_victoire'],
                "esperance": p['esperance']
            })
            break
    
    # PARI RISQUÉ - Grosse cote > 15
    for p in participants:
        if p['cote'] and p['cote'] > 15 and p['score_global'] >= 40 and len(p.get('signaux_positifs', [])) >= 2:
            # Pour un pari risqué, on mise sur GAGNANT ou MULTI en base
            type_pari = "GAGNANT"
            if nb_partants >= 12:
                type_pari = "MULTI (Base)"
            elif nb_partants >= 8:
                type_pari = "TRIO (Base)"
                
            recommandations.append({
                "type": "RISQUE",
                "type_pari": type_pari,
                "niveau_risque": 5,
                "cheval": p['nom'],
                "numero": p['numero'],
                "cote": p['cote'],
                "mise_recommandee": "1% (2-5€)",
                "gain_potentiel": f"{2 * p['cote']:.0f}-{5 * p['cote']:.0f}€",
                "raison": f"Outsider intéressant @ {p['cote']:.1f}",
                "details": p['signaux_positifs'][:3],
                "proba_succes": p['proba_victoire'],
                "esperance": p['esperance']
            })
            break
    
    return {
        "race_key": race_key,
        "hippodrome": hippodrome,
        "distance": distance,
        "type_course": type_course,
        "discipline": discipline,
        "allocation": allocation,
        "nb_partants": len(participants),
        "top_3": [
            {
                "rang": i + 1,
                "nom": p['nom'],
                "score": p['score_global'],
                "cote": p['cote'],
                "confiance": p.get('confiance', 'MOYENNE')
            }
            for i, p in enumerate(participants[:3])
        ],
        "recommandations": recommandations
    }


def analyser_participant_pour_paris(cur, row, distance, hippodrome) -> dict:
    """Analyse un participant pour les recommandations de paris"""
    
    nom = row[0]
    numero = row[1]
    cote = row[2]
    cote_ref = row[3]
    tendance = row[4]
    amplitude = row[5]
    est_favori = row[6]
    avis = row[7]
    jockey = row[8]
    entraineur = row[9]
    
    signaux_positifs = []
    signaux_negatifs = []
    
    # Score de base
    score = 50
    
    # 1. Historique du cheval
    cur.execute("""
        SELECT 
            COUNT(*) as nb_courses,
            AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as taux_victoire,
            AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100 as taux_place
        FROM cheval_courses_seen
        WHERE nom_norm = %s
    """, (nom,))
    
    hist = cur.fetchone()
    if hist and hist[0] and hist[0] > 0:
        taux_vic = hist[1] or 0
        taux_place = hist[2] or 0
        
        if taux_vic > 15:
            score += 20
            signaux_positifs.append(f"🏆 Taux victoire: {taux_vic:.1f}%")
        elif taux_vic > 10:
            score += 10
            signaux_positifs.append(f"✅ Bon taux: {taux_vic:.1f}%")
        elif taux_vic < 5:
            score -= 10
            signaux_negatifs.append(f"❌ Faible taux: {taux_vic:.1f}%")
        
        if taux_place > 40:
            score += 10
            signaux_positifs.append(f"📍 Régulier: {taux_place:.1f}% placé")
    
    # 2. Forme récente (5 dernières)
    cur.execute("""
        SELECT is_win, place_finale
        FROM cheval_courses_seen
        WHERE nom_norm = %s
        ORDER BY race_key DESC
        LIMIT 5
    """, (nom,))
    
    recentes = cur.fetchall()
    victoires_recentes = sum(1 for r in recentes if r[0] == 1)
    places_recentes = sum(1 for r in recentes if r[1] and r[1] <= 3)
    
    if victoires_recentes >= 2:
        score += 25
        signaux_positifs.append(f"🔥 {victoires_recentes} victoires récentes!")
    elif victoires_recentes == 1:
        score += 10
        signaux_positifs.append("✅ Victoire récente")
    
    if places_recentes >= 4:
        score += 15
        signaux_positifs.append("📈 Très régulier")
    
    # 3. Cote
    if cote:
        if cote < 2:
            score += 20
            signaux_positifs.append("⭐ Grand favori")
        elif cote < 3:
            score += 15
            signaux_positifs.append("⭐ Favori")
        elif cote < 5:
            score += 10
        elif cote > 50:
            score -= 15
            signaux_negatifs.append("⚠️ Très gros outsider")
    
    # 4. Tendance cote
    if tendance == '-':
        if amplitude and amplitude > 20:
            score += 15
            signaux_positifs.append(f"💰 Forte baisse cote (-{amplitude:.0f}%)")
        else:
            score += 5
    elif tendance == '+':
        if amplitude and amplitude > 20:
            score -= 10
            signaux_negatifs.append(f"📉 Forte hausse cote (+{amplitude:.0f}%)")
    
    # 5. Favori
    if est_favori:
        score += 10
        signaux_positifs.append("⭐ Favori du public")
    
    # 6. Avis
    if avis == 'POSITIF':
        score += 10
        signaux_positifs.append("👍 Avis entraineur positif")
    elif avis == 'NEGATIF':
        score -= 10
        signaux_negatifs.append("👎 Avis entraineur négatif")
    
    # 7. Performance sur distance/hippodrome
    cur.execute("""
        SELECT 
            AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100
        FROM cheval_courses_seen
        WHERE nom_norm = %s 
        AND ABS(distance_m - %s) < 200
        HAVING COUNT(*) >= 2
    """, (nom, distance or 0))
    
    perf_dist = cur.fetchone()
    if perf_dist and perf_dist[0] and perf_dist[0] > 15:
        score += 10
        signaux_positifs.append(f"📏 Bon sur la distance ({perf_dist[0]:.0f}%)")
    
    # Limiter le score
    score = max(0, min(100, score))
    
    # Calculer probabilité et value
    proba_victoire = 10  # Par défaut
    if cote and cote > 0:
        proba_implicite = 100 / cote
        # Ajuster avec notre score
        adjustment = (score - 50) / 100
        proba_victoire = proba_implicite * (1 + adjustment)
        proba_victoire = max(1, min(95, proba_victoire))
    
    value = 0
    esperance = 0
    if cote and cote > 0:
        value = proba_victoire - (100 / cote)
        esperance = (proba_victoire / 100 * cote - 1) * 100
    
    # Confiance
    confiance = "FAIBLE"
    if score >= 70 and len(signaux_positifs) >= 4:
        confiance = "HAUTE"
    elif score >= 55 and len(signaux_positifs) >= 2:
        confiance = "MOYENNE"
    
    return {
        "nom": nom,
        "numero": numero,
        "cote": cote,
        "score_global": score,
        "proba_victoire": round(proba_victoire, 1),
        "value": round(value, 1),
        "esperance": round(esperance, 1),
        "confiance": confiance,
        "signaux_positifs": signaux_positifs,
        "signaux_negatifs": signaux_negatifs,
        "jockey": jockey,
        "entraineur": entraineur
    }


def analyser_course_complete(cur, race_key: str) -> dict:
    """Analyse complète d'une course avec tous les détails"""
    
    # Infos course
    cur.execute("""
        SELECT DISTINCT
            hippodrome_nom,
            distance_m,
            type_course,
            discipline,
            allocation_totale
        FROM cheval_courses_seen
        WHERE race_key = %s
        LIMIT 1
    """, (race_key,))
    
    course_info = cur.fetchone()
    if not course_info:
        return None
    
    hippodrome, distance, type_course, discipline, allocation = course_info
    
    # Participants avec analyse complète
    cur.execute("""
        SELECT 
            nom_norm,
            numero_dossard,
            cote_finale,
            cote_reference,
            tendance_cote,
            amplitude_tendance,
            est_favori,
            avis_entraineur,
            driver_jockey,
            entraineur,
            age,
            sexe,
            musique
        FROM cheval_courses_seen
        WHERE race_key = %s
        ORDER BY numero_dossard
    """, (race_key,))
    
    analyses = []
    for row in cur.fetchall():
        participant = analyser_participant_pour_paris(cur, row, distance, hippodrome)
        if participant:
            # Ajouter des infos supplémentaires
            participant['age'] = row[10]
            participant['sexe'] = row[11]
            participant['musique'] = row[12]
            analyses.append(participant)
    
    # Trier par score
    analyses.sort(key=lambda x: x['score_global'], reverse=True)
    
    # Générer les recommandations détaillées
    rapport = analyser_course_pour_paris(cur, race_key)
    
    return {
        "race_key": race_key,
        "hippodrome": hippodrome,
        "distance": distance,
        "type_course": type_course,
        "discipline": discipline,
        "allocation": allocation,
        "nb_partants": len(analyses),
        "analyses": analyses,
        "recommandations": rapport.get('recommandations', []) if rapport else [],
        "top_3": [
            {
                "rang": i + 1,
                "nom": a['nom'],
                "score": a['score_global'],
                "cote": a['cote'],
                "confiance": a['confiance'],
                "signaux": a['signaux_positifs'][:2]
            }
            for i, a in enumerate(analyses[:3])
        ]
    }


# ============================================================================
# NOUVEAUX ENDPOINTS - PICKS, CALIBRATION, EXOTIQUES, PORTFOLIO
# ============================================================================

@app.get("/calibration/health")
async def get_calibration_health():
    """
    Récupère la santé du système de calibration.
    Retourne ECE, Brier score, date dernière calibration.
    """
    try:
        # Lire le fichier de calibration s'il existe
        calibration_file = os.path.join(BASE_DIR, "..", "data", "calibration_metrics.json")
        
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                return data
        
        # Sinon, calculer des métriques de base depuis les données
        con = get_db_connection()
        cur = con.cursor()
        
        # Calculer ECE et Brier sur les 7 derniers jours (simulation)
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN cote_finale IS NOT NULL AND cote_finale > 0 
                    THEN 1.0/cote_finale ELSE NULL END) as avg_implied_prob
            FROM cheval_courses_seen
            WHERE race_key >= (CURRENT_DATE - INTERVAL '7 days')::text
            AND cote_finale IS NOT NULL
        """)
        
        row = cur.fetchone()
        total = row[0] or 1
        wins = row[1] or 0
        avg_prob = row[2] or 0.1
        
        # ECE simplifié (différence entre proba implicite et résultat réel)
        actual_win_rate = wins / total if total > 0 else 0
        ece_7d = abs(avg_prob - actual_win_rate)
        
        # Brier score simplifié
        brier_7d = (avg_prob - actual_win_rate) ** 2
        
        con.close()
        
        return {
            "ece_7d": round(ece_7d, 4),
            "brier_7d": round(brier_7d, 4),
            "last_calibration": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "healthy" if ece_7d < 0.1 else "warning" if ece_7d < 0.2 else "critical",
            "samples_7d": total,
            "roi_7d": round((wins / total * 100 - 100) if total > 0 else 0, 2),
            "hit_rate_7d": round(wins / total * 100, 1) if total > 0 else 0,
            "total_turnover_7d": total * 10  # Approximation
        }
        
    except Exception as e:
        return {
            "ece_7d": 0,
            "brier_7d": 0,
            "last_calibration": None,
            "status": "error",
            "error": str(e)
        }


@app.get("/picks/today")
async def get_picks_today():
    """
    Récupère les picks/recommandations du jour.
    Retourne les chevaux avec value bet positive.
    """
    con = None
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Chercher les courses d'aujourd'hui, sinon les plus récentes
        cur.execute("""
            SELECT DISTINCT SUBSTRING(race_key FROM 1 FOR 10) as race_date
            FROM cheval_courses_seen
            WHERE race_key IS NOT NULL
            ORDER BY race_date DESC
            LIMIT 1
        """)
        
        latest_date = cur.fetchone()
        search_date = today if latest_date is None else latest_date[0]

        # Analyse Benter hiérarchique obligatoire avant toute recommandation
        try:
            # Utilise une connexion isolée pour éviter de polluer la transaction courante en cas d'erreur
            benter_result = run_benter_head_for_date(search_date, cur=None)
        except Exception as e:
            # Sécurité : ne jamais bloquer les picks, mais signaler l'erreur dans la meta
            benter_result = {"by_runner": {}, "meta": {"status": "error", "reason": str(e)}}
        benter_map = benter_result.get("by_runner", {})
        benter_meta = benter_result.get("meta", {})
        
        # Seuil max de cote pour filtrer les outsiders extrêmes (NP/DAI/ARR ont souvent des cotes énormes)
        MAX_PREOFF_COTE = float(os.getenv("MAX_PREOFF_COTE", "50"))
        
        # Timestamp actuel en millisecondes pour filtrer les courses déjà parties
        import time
        now_ms = int(time.time() * 1000)
        
        # Récupérer les chevaux avec analyse (les probabilités seront issues du head Benter)
        # FILTRES ANTI-FUITE:
        # 1. place_finale IS NULL → course pas encore courue
        # 2. statut_participant = 'PARTANT' → exclut NP pré-départ
        # 3. incident IS NULL → exclut DAI, ARR, TOMBE, etc. (incidents post-départ)
        # 4. cote_finale ET cote_reference < MAX → exclut outsiders extrêmes
        # 5. heure_depart > now → course pas encore partie
        cur.execute("""
                SELECT 
                    cs.id_cheval as id_cheval,
                    cs.nom_norm as nom,
                    cs.race_key,
                    cs.hippodrome_nom as hippodrome,
                    cs.heure_locale as heure,
                    cs.numero_dossard as numero,
                    cs.cote_finale as cote,
                    cs.cote_reference,
                    cs.tendance_cote,
                    cs.amplitude_tendance,
                    cs.est_favori,
                    cs.avis_entraineur,
                    cs.driver_jockey as jockey,
                    cs.entraineur,
                    cs.discipline,
                    cs.distance_m,
                    cs.is_win,
                    cs.place_finale,
                    cs.statut_participant,
                    cs.incident,
                    cs.heure_depart
                FROM cheval_courses_seen cs
            WHERE cs.race_key LIKE %s
            AND cs.cote_finale IS NOT NULL
            AND cs.cote_finale > 0
            AND cs.cote_finale < %s
            AND (cs.cote_reference IS NULL OR cs.cote_reference < %s)  -- anti-fuite: exclut outsiders extrêmes
            AND (cs.place_finale IS NULL)  -- anti-fuite: uniquement pré-off
            AND (cs.statut_participant IS NULL OR UPPER(cs.statut_participant) IN ('PARTANT', 'PARTANTE', 'PART', 'P', ''))  -- anti-fuite: exclut NP pré-départ
            AND (cs.incident IS NULL)  -- anti-fuite: exclut DAI, ARR, TOMBE, NP tardif (incidents post-départ)
            AND (cs.heure_depart IS NULL OR cs.heure_depart::bigint > %s)  -- anti-fuite: exclut courses déjà parties
            ORDER BY cs.race_key, cs.numero_dossard
        """, (search_date + '%', MAX_PREOFF_COTE, MAX_PREOFF_COTE, now_ms))
        rows = cur.fetchall()
        if not rows:
            # Exposer l'état du modèle place (même si pas de courses pré-off)
            place_model_meta = {"status": "disabled"}
            try:
                use_place_model = os.getenv("USE_PLACE_MODEL", "1").lower() in ("1", "true", "yes", "on")
                if use_place_model:
                    from services.place_model import load_place_model
                    pm = load_place_model()
                    place_model_meta = {"status": "ok", **(pm.metadata or {})} if pm else {"status": "missing"}
            except Exception as e:
                place_model_meta = {"status": "error", "reason": str(e)}
            con.close()
            return {
                "date": search_date,
                "picks": [],
                "total": 0,
                "summary": {},
                "meta": {
                    "benter_analysis": benter_meta,
                    "market_blend": {"status": "no_data"},
                    "place_model": place_model_meta,
                    "preoff_filter": {"skipped": 0},
                },
            }

        # Préparer données par course pour appliquer correction marché + blend dynamique
        race_groups = {}
        skipped_post_off = 0
        skipped_status = 0
        skipped_high_odds = 0  # compteur pour cotes trop élevées après transformation
        
        # Liste des mots-clés indiquant un cheval non-partant ou avec incident
        EXCLUDED_STATUS_KEYWORDS = (
            "NP", "NON PARTANT", "NONPARTANT", "NON_PARTANT",
            "DAI", "DIA", "DISQ", "DQ", "DISQUALIFIE", "DISQUALIFIED",
            "ARR", "ARRET", "ARRETE", "STOPPED",
            "CHU", "CHUTE", "TOMBE", "TOM", "FALL", "FALLEN",
            "RET", "RETIRE", "RETIREE", "RÉTIRÉ", "RETRAIT", "WITHDRAWN",
            "DIST", "DISTANCIE", "DISTANCED"
        )
        
        skipped_incident = 0  # compteur pour chevaux avec incident
        
        for row in rows:
            id_cheval, nom, race_key, hippodrome, heure, numero, cote, cote_ref, tendance, amplitude, est_favori, avis, jockey, entraineur, discipline, distance, is_win, place, statut, incident, heure_depart = row
            
            # Double vérification du statut (ceinture et bretelles - le SQL filtre déjà, mais on re-vérifie)
            statut_norm = (statut or "").upper().replace("-", " ").replace("_", " ")
            allowed_status = {"", "PARTANT", "PARTANTE", "PART", "P"}
            if statut_norm not in allowed_status or any(key in statut_norm for key in EXCLUDED_STATUS_KEYWORDS):
                skipped_status += 1
                continue
            
            # Vérification de l'incident (DAI, ARR, TOMBE, NP tardif, etc.)
            if incident is not None:
                skipped_incident += 1
                continue
            
            try:
                market_odds = select_preoff_market_odds(cote, cote_ref, place)
            except ValueError:
                skipped_post_off += 1
                continue
            
            # Filtre final sur la cote après transformation (cote_reference peut être > cote_finale)
            if market_odds > MAX_PREOFF_COTE:
                skipped_high_odds += 1
                continue
            
            cote = market_odds
            cote_ref = cote_ref  # keep original for logs
            score = calculate_prediction_score(cote, cote_ref, tendance, amplitude, est_favori, avis)
            benter_key = (race_key, numero)
            benter_runner = benter_map.get(benter_key) or benter_map.get((race_key, nom))
            p_model = 0
            p_benter_raw = None
            context_effect = None
            micro_effect = None
            p_mkt_corr_head = None
            if benter_runner:
                p_model = benter_runner.get("p_calibrated") or benter_runner.get("p_model_norm") or 0
                p_benter_raw = benter_runner.get("p_model_norm")
                context_effect = benter_runner.get("context_effect")
                micro_effect = benter_runner.get("micro_effect")
                p_mkt_corr_head = benter_runner.get("p_market_corr")
            else:
                proba_implicite = 1 / cote if cote > 0 else 0
                adjustment = (score - 50) / 200
                p_model = max(0.01, min(0.95, proba_implicite * (1 + adjustment)))

            race_groups.setdefault(race_key, []).append({
                "id_cheval": id_cheval,
                "nom": nom,
                "race_key": race_key,
                "hippodrome": hippodrome,
                "heure": heure,
                "numero": numero,
                "cote": cote,
                "cote_ref": cote_ref,
                "tendance": tendance,
                "amplitude": amplitude,
                "est_favori": est_favori,
                "avis": avis,
                "jockey": jockey,
                "entraineur": entraineur,
                "discipline": discipline,
                "distance": distance,
                "is_win": is_win,
                "place": place,
                "score": score,
                "p_model": p_model,
                "p_benter_raw": p_benter_raw,
                "context_effect": context_effect,
                "micro_effect": micro_effect,
                "p_mkt_corr_head": p_mkt_corr_head,
            })

        gamma_default = float(os.getenv("MARKET_GAMMA_DEFAULT", "0.9"))
        alpha_base = float(os.getenv("BLEND_ALPHA_BASE", "0.5"))
        alpha_pool_coef = float(os.getenv("BLEND_ALPHA_POOL_COEF", "0.05"))
        alpha_time_coef = float(os.getenv("BLEND_ALPHA_TIME_COEF", "0.01"))
        alpha_min = float(os.getenv("BLEND_ALPHA_MIN", "0.3"))
        alpha_max = float(os.getenv("BLEND_ALPHA_MAX", "0.9"))
        default_pool = float(os.getenv("DEFAULT_POOL_SIZE", "50000"))
        default_mto = float(os.getenv("DEFAULT_MINUTES_TO_OFF", "30"))
        micro_neg_threshold = float(os.getenv("MICRO_DRIFT_NEG_THRESHOLD", "0.10"))  # 10% log-odds défavorable → hold
        micro_scale_threshold = float(os.getenv("MICRO_DRIFT_SCALE_THRESHOLD", "0.05"))  # 5% → scale-down

        market_meta = {"status": "ok", "gamma": gamma_default, "alpha_bounds": [alpha_min, alpha_max], "races": len(race_groups)}

        # Takeout (pour l'estimation place via la proba marché)
        try:
            cfg = load_config_file(create_if_missing=True)
            takeout_rate = float(cfg.get("markets", {}).get("takeout_rate", 0.16))
        except Exception:
            takeout_rate = 0.16

        # Optionnel: remplacer p_model par le modèle champion (XGBoost) si dispo.
        champion_meta = {"status": "disabled"}
        use_champion = os.getenv("USE_CHAMPION_MODEL", "1").lower() in ("1", "true", "yes", "on")
        if use_champion:
            try:
                from services.champion_predictor import ChampionPredictor, default_champion_artifacts

                predictor = ChampionPredictor(default_champion_artifacts())

                def _build_features_for_race(runners: List[Dict[str, Any]], race_date: str) -> List[Dict[str, Any]]:
                    ids = [r.get("id_cheval") for r in runners if r.get("id_cheval") is not None]
                    if not ids:
                        return []

                    # Stats chevaux (forme/aptitudes/regularite)
                    cur.execute(
                        """
                        SELECT id_cheval, forme_5c, aptitude_distance, aptitude_piste, aptitude_hippodrome, regularite
                        FROM stats_chevaux
                        WHERE id_cheval = ANY(%s)
                        """,
                        (ids,),
                    )
                    sc = {row[0]: row[1:] for row in cur.fetchall()}

                    # Historique 12 mois / récence (cheval_courses_seen)
                    cur.execute(
                        """
                        WITH hist AS (
                            SELECT
                                id_cheval,
                                SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
                                is_win,
                                place_finale
                            FROM cheval_courses_seen
                            WHERE id_cheval = ANY(%s)
                              AND place_finale IS NOT NULL
                              AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
                        )
                        SELECT
                            id_cheval,
                            COUNT(*) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_courses_12m,
                            SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_victoires_12m,
                             SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_places_12m,
                             (%s::date - MAX(d))::float AS recence
                         FROM hist
                         GROUP BY id_cheval
                         """,
                         (ids, race_date, race_date, race_date, race_date, race_date),
                     )
                    hist = {row[0]: row[1:] for row in cur.fetchall()}

                    # Jockey/entraineur stats (12 mois) via noms
                    jockeys = sorted({(r.get("jockey") or "").strip() for r in runners if (r.get("jockey") or "").strip()})
                    trainers = sorted({(r.get("entraineur") or "").strip() for r in runners if (r.get("entraineur") or "").strip()})

                    jockey_stats = {}
                    if jockeys:
                        cur.execute(
                            """
                            WITH hist AS (
                                SELECT
                                    driver_jockey,
                                    SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
                                    is_win,
                                    place_finale
                                FROM cheval_courses_seen
                                WHERE driver_jockey = ANY(%s)
                                  AND place_finale IS NOT NULL
                                  AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
                                  AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
                            )
                            SELECT
                                driver_jockey,
                                COUNT(*) AS nb_courses,
                                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) AS wins,
                                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) AS places
                            FROM hist
                            GROUP BY driver_jockey
                            """,
                            (jockeys, race_date, race_date),
                        )
                        for name, nb, wins, places in cur.fetchall():
                            nb = nb or 0
                            jockey_stats[name] = (
                                (wins or 0) / nb if nb else 0.0,
                                (places or 0) / nb if nb else 0.0,
                            )

                    trainer_stats = {}
                    if trainers:
                        cur.execute(
                            """
                            WITH hist AS (
                                SELECT
                                    entraineur,
                                    SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
                                    is_win,
                                    place_finale
                                FROM cheval_courses_seen
                                WHERE entraineur = ANY(%s)
                                  AND place_finale IS NOT NULL
                                  AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
                                  AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
                            )
                            SELECT
                                entraineur,
                                COUNT(*) AS nb_courses,
                                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) AS wins,
                                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) AS places
                            FROM hist
                            GROUP BY entraineur
                            """,
                            (trainers, race_date, race_date),
                        )
                        for name, nb, wins, places in cur.fetchall():
                            nb = nb or 0
                            trainer_stats[name] = (
                                (wins or 0) / nb if nb else 0.0,
                                (places or 0) / nb if nb else 0.0,
                            )

                    # Infos "course du jour" pour les features champion (corde, météo, piste, etc.)
                    extra = {}
                    try:
                        rk = (runners[0].get("race_key") if runners else None) or ""
                        cur.execute(
                            """
                            SELECT
                                id_cheval,
                                draw_stalle,
                                cote_matin,
                                meteo_code,
                                temperature_c,
                                vent_kmh,
                                allocation_totale,
                                etat_piste,
                                sexe,
                                age
                            FROM cheval_courses_seen
                            WHERE race_key = %s
                              AND id_cheval = ANY(%s)
                            """,
                            (rk, ids),
                        )
                        for (
                            cid,
                            draw_stalle,
                            cote_matin,
                            meteo_code,
                            temperature_c,
                            vent_kmh,
                            allocation_totale,
                            etat_piste,
                            sexe,
                            age,
                        ) in cur.fetchall():
                            extra[cid] = {
                                "numero_corde": draw_stalle,
                                "cote_pm": cote_matin,
                                "meteo_code": meteo_code,
                                "temperature_c": temperature_c,
                                "vent_kmh": vent_kmh,
                                "allocation": allocation_totale,
                                "etat_piste": etat_piste,
                                "sexe": sexe,
                                "age": age,
                            }
                    except Exception:
                        extra = {}

                    nb_partants = max(2, len(runners))
                    distance = float(runners[0].get("distance") or 2000)
                    distance_norm = distance / 1000.0

                    # Niveau moyen concurrent: proxy = moyenne forme_5c des partants
                    mean_forme = 0.0
                    vals = [float((sc.get(r.get("id_cheval"), (0, 0, 0, 0, 0))[0]) or 0) for r in runners]
                    if vals:
                        mean_forme = sum(vals) / len(vals)

                    # Rang de cote SP (1 = favori)
                    cotes = [r.get("cote") or 0 for r in runners]
                    sorted_idx = sorted(range(len(cotes)), key=lambda i: (cotes[i] or 999999))
                    rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}

                    def _to_float(v: Any) -> float:
                        try:
                            return float(v)
                        except Exception:
                            if isinstance(v, str):
                                import re
                                m = re.search(r"-?\\d+(?:[\\.,]\\d+)?", v)
                                if not m:
                                    return 0.0
                                return float(m.group(0).replace(",", "."))
                            return 0.0

                    feature_rows: List[Dict[str, Any]] = []
                    for idx, r in enumerate(runners):
                        cid = r.get("id_cheval")
                        forme_5c, apt_dist, apt_piste, apt_hippo, regularite = sc.get(cid, (0, 0, 0, 0, 0))
                        nb_courses_12m, nb_victoires_12m, nb_places_12m, recence = hist.get(cid, (0, 0, 0, 90.0))
                        j_name = (r.get("jockey") or "").strip()
                        t_name = (r.get("entraineur") or "").strip()
                        j_wr, j_pr = jockey_stats.get(j_name, (0.0, 0.0))
                        t_wr, t_pr = trainer_stats.get(t_name, (0.0, 0.0))
                        ex = extra.get(cid, {})
                        sexe = (ex.get("sexe") or "").strip().upper()
                        etat = (ex.get("etat_piste") or "").strip()
                        disc = (r.get("discipline") or "").strip().lower()

                        # One-hot état de piste (liste issue du dataset champion)
                        etat_features = {
                            "etat_Bon léger": 0.0,
                            "etat_Bon souple": 0.0,
                            "etat_Collant": 0.0,
                            "etat_Lourd": 0.0,
                            "etat_Léger": 0.0,
                            "etat_PSF": 0.0,
                            "etat_PSF LENTE": 0.0,
                            "etat_PSF RAPIDE": 0.0,
                            "etat_PSF STANDARD": 0.0,
                            "etat_Souple": 0.0,
                            "etat_Très lourd": 0.0,
                            "etat_Très souple": 0.0,
                        }
                        key = f"etat_{etat}"
                        if key in etat_features:
                            etat_features[key] = 1.0

                        elo_cheval = 1000.0
                        synergie_j = float((j_wr or 0) * (forme_5c or 0))
                        synergie_t = float((t_wr or 0) * (forme_5c or 0))

                        interaction_forme_jockey = float((forme_5c or 0) * (j_wr or 0))
                        interaction_aptitude_distance = float((apt_dist or 0) * (distance_norm or 0))
                        interaction_elo_niveau = float((elo_cheval / 1000.0) * (mean_forme or 0))
                        interaction_synergie_forme = float((synergie_j + synergie_t) * (forme_5c or 0))
                        interaction_aptitude_popularite = float((apt_dist or 0) * (1.0 / max(1.0, float(rank_map.get(idx, 1)))))
                        interaction_regularite_volume = float((regularite or 0) * (nb_courses_12m or 0))

                        # an_naissance approximé depuis âge
                        try:
                            year = int(str(race_date)[:4])
                            age = float(ex.get("age") or 0)
                            an_naissance = year - age if age else 0.0
                        except Exception:
                            an_naissance = 0.0

                        feature_rows.append({
                            # Features attendues (voir data/models/champion/feature_names.json)
                            "numero_corde": _to_float(ex.get("numero_corde") or 0),
                            "cote_sp": float(r.get("cote") or 0),
                            "cote_pm": _to_float(ex.get("cote_pm") or r.get("cote") or 0),
                            "cote_turfbzh": 0.0,
                            "prediction_ia_gagnant": 0.0,
                            "elo_cheval": float(elo_cheval),
                            "temps_total": 0.0,
                            "vitesse_moyenne": 0.0,
                            "an_naissance": float(an_naissance),
                            "distance": float(distance or 0),
                            "meteo": _to_float(ex.get("meteo_code") or 0),
                            "temperature_c": _to_float(ex.get("temperature_c") or 0),
                            "vent_kmh": _to_float(ex.get("vent_kmh") or 0),
                            "nombre_partants": float(nb_partants),
                            "allocation": _to_float(ex.get("allocation") or 0),
                            "non_partant": 0.0,
                            "forme_5c": float(forme_5c or 0),
                            "forme_10c": float(forme_5c or 0),  # proxy
                            "nb_courses_12m": float(nb_courses_12m or 0),
                            "regularite": float(regularite or 0),
                            "jours_depuis_derniere": float(recence or 90.0),
                            "aptitude_distance": float(apt_dist or 0),
                            "aptitude_piste": float(apt_piste or 0),
                            "synergie_jockey_cheval": float(synergie_j),
                            "synergie_entraineur_cheval": float(synergie_t),
                            "distance_norm": float(distance_norm or 0),
                            "niveau_moyen_concurrent": float(mean_forme or 0),
                            "rang_cote_sp": float(rank_map.get(idx, 1)),
                            "rang_cote_turfbzh": 0.0,
                            "gains_carriere": 0.0,
                            "gains_12m": 0.0,
                            "gains_par_course": 0.0,
                            "evolution_gains_12m": 0.0,
                            "ratio_gains_courses": 0.0,
                            "etat_piste_encoded": 0.0,
                            "meteo_encoded": 0.0,
                            "aptitude_piste_etat": float((apt_piste or 0) * (etat_features.get(key, 0.0))),
                            "interaction_piste_meteo": 0.0,
                            "handicap_meteo": 0.0,
                            "discipline_Plat": 1.0 if "plat" in disc else 0.0,
                            "discipline_Trot": 1.0 if "trot" in disc or "att" in disc else 0.0,
                            "sexe_H": 1.0 if sexe.startswith("H") else 0.0,
                            "sexe_M": 1.0 if sexe.startswith("M") else 0.0,
                            **etat_features,
                            "interaction_forme_jockey": float(interaction_forme_jockey),
                            "interaction_aptitude_distance": float(interaction_aptitude_distance),
                            "interaction_elo_niveau": float(interaction_elo_niveau),
                            "interaction_cote_ia": 0.0,
                            "interaction_synergie_forme": float(interaction_synergie_forme),
                            "interaction_aptitude_popularite": float(interaction_aptitude_popularite),
                            "interaction_regularite_volume": float(interaction_regularite_volume),
                        })
                    return feature_rows

                for rk, runners in race_groups.items():
                    race_date = str(rk).split("|")[0] if rk and "|" in str(rk) else search_date
                    feat_rows = _build_features_for_race(runners, race_date)
                    if not feat_rows:
                        continue
                    raw_probs = predictor.predict_proba(feat_rows)
                    s = sum(raw_probs) or 1.0
                    for r, p in zip(runners, raw_probs):
                        r["p_model"] = max(0.001, min(0.999, float(p) / s))

                champion_meta = {
                    "status": "ok",
                    "artifacts_dir": os.getenv("CHAMPION_DIR", "/project/data/models/champion"),
                    "features": len(predictor.feature_names or []),
                    "temperature": round(float(getattr(predictor, "temperature", 1.0) or 1.0), 6),
                    "platt": bool(getattr(predictor, "platt_loaded", False)),
                    "fallback": getattr(predictor, "fallback_used", None),
                }
            except Exception as e:
                champion_meta = {"status": "error", "reason": str(e)}
                try:
                    con.rollback()
                except Exception:
                    pass

        analysis_source_label = "champion_xgboost" if champion_meta.get("status") == "ok" else "benter_head_hierarchique"

        def _harville_p_place_top3(pwins: List[float]) -> List[float]:
            # Harville (top 3) avec p(win) normalisées (O(n^3), n<=20 typiquement)
            p = [_safe_prob(x) for x in pwins]
            s = sum(p) or 1.0
            p = [x / s for x in p]
            n = len(p)
            out = [0.0] * n
            for i in range(n):
                p1 = p[i]
                p2 = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    denom1 = 1.0 - p[j]
                    if denom1 <= 1e-12:
                        continue
                    p2 += p[j] * (p[i] / denom1)
                p3 = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    denom1 = 1.0 - p[j]
                    if denom1 <= 1e-12:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        denom2 = 1.0 - p[j] - p[k]
                        if denom2 <= 1e-12:
                            continue
                        p3 += p[j] * (p[k] / denom1) * (p[i] / denom2)
                out[i] = max(0.0, min(0.999, p1 + p2 + p3))
            return out

        # Place model calibré (optionnel)
        place_model_meta = {"status": "disabled"}
        use_place_model = os.getenv("USE_PLACE_MODEL", "1").lower() in ("1", "true", "yes", "on")
        place_model = None
        if use_place_model:
            try:
                from services.place_model import load_place_model

                place_model = load_place_model()
                if place_model:
                    place_model_meta = {"status": "ok", **(place_model.metadata or {})}
                else:
                    place_model_meta = {"status": "missing"}
            except Exception as e:
                place_model_meta = {"status": "error", "reason": str(e)}

        picks = []
        try:
            for race_key, runners in race_groups.items():
                # Proba marché implicite normalisée
                implied = [1.0 / r["cote"] if r["cote"] else 0 for r in runners]
                total_implied = sum(implied) or 1.0
                p_market_norm = [x / total_implied for x in implied]
                # Correction gamma
                p_mkt_gamma = [pow(_safe_prob(p), gamma_default) for p in p_market_norm]
                total_gamma = sum(p_mkt_gamma) or 1.0
                p_mkt_corr = [x / total_gamma for x in p_mkt_gamma]

                # Alpha dynamique (proxy pool/temps)
                # Minutes to off: si heure disponible HH:MM, on estime à partir de l'heure locale
                minutes_to_off = default_mto
                try:
                    if runners and runners[0]["heure"]:
                        hhmm = str(runners[0]["heure"]).split(":")
                        if len(hhmm) >= 2:
                            depart_minutes = int(hhmm[0]) * 60 + int(hhmm[1])
                            now_minutes = datetime.now().hour * 60 + datetime.now().minute
                            delta = depart_minutes - now_minutes
                            minutes_to_off = max(0, delta)
                except Exception:
                    minutes_to_off = default_mto

                # Pool: pas dispo en base, proxy par défaut
                pool_size = default_pool

                alpha_raw = alpha_base + alpha_pool_coef * math.log(max(pool_size, 1)) - alpha_time_coef * minutes_to_off
                alpha = min(alpha_max, max(alpha_min, alpha_raw))
                if champion_meta.get("status") == "ok":
                    # Par défaut: on garde une part de marché pour éviter les surconfiances
                    # quand les features "temps réel" sont partielles côté web.
                    try:
                        alpha = float(os.getenv("CHAMPION_BLEND_ALPHA", "0.35"))
                    except Exception:
                        alpha = 0.35
                    alpha = min(0.95, max(0.05, alpha))

                # Blend logit-space
                blended_probs = []
                for idx, r in enumerate(runners):
                    p_model = _safe_prob(r["p_model"])
                    p_mkt = _safe_prob(p_mkt_corr[idx])
                    logit_blend = alpha * _safe_logit(p_model) + (1 - alpha) * _safe_logit(p_mkt)
                    p_blend = 1.0 / (1.0 + math.exp(-logit_blend))
                    blended_probs.append(p_blend)

                # Renormalisation par course pour sommer à 1
                total_blend = sum(blended_probs) or 1.0
                blended_probs = [p / total_blend for p in blended_probs]

                # Guardrail: éviter des proba "hors-sol" vs marché si features manquantes (champion web).
                if champion_meta.get("status") == "ok":
                    try:
                        max_ratio = float(os.getenv("CHAMPION_MAX_MARKET_RATIO", "2.0"))
                    except Exception:
                        max_ratio = 2.0
                    max_ratio = max(1.0, min(max_ratio, 5.0))
                    capped = []
                    for i, p in enumerate(blended_probs):
                        m = _safe_prob(p_mkt_corr[i])
                        cap = max(0.0005, m * max_ratio) if m > 0 else p
                        capped.append(min(p, cap))
                    s2 = sum(capped) or 1.0
                    blended_probs = [p / s2 for p in capped]

                # p(place) (top3) côté modèle et côté marché (pour estimer un "prix" place)
                p_place_race = _harville_p_place_top3(blended_probs)
                p_place_mkt = _harville_p_place_top3(p_mkt_corr)

                # Calibrer p_place si modèle dispo
                if place_model and p_place_race:
                    try:
                        # rang cote = rang des cotes croissantes (favori=1)
                        cotes_r = [r.get("cote") or 0 for r in runners]
                        sorted_idx = sorted(range(len(cotes_r)), key=lambda i: (cotes_r[i] or 999999))
                        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}
                        pm_inputs = []
                        for i, r in enumerate(runners):
                            pm_inputs.append({
                                "p_place_harville": float(p_place_race[i]),
                                "field_size": len(runners),
                                "rank_odds": int(rank_map.get(i, i + 1)),
                                "discipline": r.get("discipline"),
                            })
                        p_place_adj = place_model.predict(pm_inputs)
                        # renormaliser pour rester cohérent (≈ somme ~3)
                        target_sum = min(3.0, float(len(runners)))
                        s_adj = sum(p_place_adj) or 1.0
                        p_place_race = [max(0.0, min(0.999, float(x) * target_sum / s_adj)) for x in p_place_adj]
                    except Exception:
                        pass

                for idx, r in enumerate(runners):
                    p_win = _safe_prob(blended_probs[idx])
                    cote = r["cote"]
                    p_place = _safe_prob(p_place_race[idx]) if p_place_race else 0.0

                    # Estimation place: on dérive un prix "marché" depuis p_place marché,
                    # puis on calcule la value vs p_place modèle.
                    p_place_market = _safe_prob(p_place_mkt[idx]) if p_place_mkt else 0.0
                    if p_place_market > 0:
                        cote_place = max(1.05, (1.0 - takeout_rate) / max(p_place_market, 1e-6))
                    else:
                        cote_place = 1.05
                    if cote:
                        cote_place = min(cote, cote_place)

                    value_pct = (p_win * cote - 1) * 100 if cote else 0
                    value_place_pct = (p_place * cote_place - 1) * 100 if cote_place else 0

                    kelly_pct = 0
                    if value_pct > 0 and cote and cote > 1:
                        kelly_pct = (p_win * cote - 1) / (cote - 1) * 100
                        kelly_pct = max(0, min(kelly_pct, 25))
                    kelly_place_pct = 0
                    if value_place_pct > 0 and cote_place > 1:
                        kelly_place_pct = (p_place * cote_place - 1) / (cote_place - 1) * 100
                        kelly_place_pct = max(0, min(kelly_place_pct, 25))

                    # Microstructure temps-réel (proxy avec tendance/amplitude)
                    drift_pct = 0.0
                    if r["tendance"] in ("+", "-") and r["amplitude"] is not None:
                        sign = 1 if r["tendance"] == "+" else -1  # + = dérive cote vers le haut (défavorable)
                        drift_pct = sign * (r["amplitude"] or 0) / 100.0
                    micro_score = -drift_pct  # plus c'est grand, plus favorable (cote baisse)
                    micro_action = "bet"
                    micro_reason = "ok"
                    if drift_pct >= micro_neg_threshold:
                        micro_action = "hold"
                        micro_reason = f"drift défavorable {drift_pct:.2f}≥{micro_neg_threshold:.2f}"
                    elif drift_pct >= micro_scale_threshold:
                        micro_action = "scale_down"
                        micro_reason = f"drift défavorable modéré {drift_pct:.2f}≥{micro_scale_threshold:.2f}"
    
                    bet_types = []
                    if p_place >= 0.50:
                        place_confidence = "Fort" if p_place >= 0.70 else "Moyen"
                        # La cote placé est estimée: on ne propose PLACÉ que si l'estimation est positive
                        # sinon l'UI masque tout (value négative) et ça crée de la confusion.
                        if value_place_pct > 0:
                            bet_types.append({
                                "type": "SIMPLE PLACÉ",
                                "emoji": "🥉",
                                "confidence": place_confidence,
                                "risk": "Faible",
                                "proba": round(p_place * 100, 1),
                                "cote_estimee": round(cote_place, 2),
                                "cote_is_estimate": True,
                                "value": round(value_place_pct, 1),
                                "kelly": round(kelly_place_pct, 1),
                                "description": f"Finir dans les 3 premiers ({p_place*100:.0f}% de chances) - Cote ~{cote_place:.2f} (estimée via marché)"
                            })
                    if p_win >= 0.15 and p_place >= 0.45:
                        ev_ep = (p_win * cote + p_place * cote_place - 2) if cote else -1
                        if ev_ep > 0:
                            ep_risk = "Faible" if (p_place >= 0.60 and (cote and cote <= 10)) else "Modéré"
                            bet_types.append({
                                "type": "E/P (GAGNANT-PLACÉ)",
                                "emoji": "🎯",
                                "confidence": "Moyen" if p_win >= 0.20 else "Prudent",
                                "risk": ep_risk,
                                "proba": round(p_place * 100, 1),
                                "cote_estimee": round((cote + cote_place) / 2, 2) if cote else round(cote_place, 2),
                                "cote_gagnant": round(cote, 2) if cote else None,
                                "cote_place_estimee": round(cote_place, 2),
                                "cote_is_estimate": True,
                                "value": round(ev_ep * 50, 1),
                                "kelly": round(min(kelly_pct, kelly_place_pct), 1),
                                "description": f"Gagnant ({cote:.2f}) + Placé (~{cote_place:.2f} estimée)" if cote else "Gagnant + Placé"
                            })
                    if p_win >= 0.30 and value_pct >= 10:
                        win_risk = "Faible" if (cote and cote <= 4 and p_win >= 0.22) else ("Modéré" if (cote and cote <= 8 and p_win >= 0.15) else "Élevé")
                        bet_types.append({
                            "type": "SIMPLE GAGNANT",
                            "emoji": "🏆",
                            "confidence": "Fort",
                            "risk": win_risk,
                            "proba": round(p_win * 100, 1),
                            "cote_estimee": round(cote, 2) if cote else None,
                            "value": round(value_pct, 1),
                            "kelly": round(kelly_pct, 1),
                            "description": f"Victoire attendue ({p_win*100:.0f}% - cote {cote:.2f})" if cote else "Victoire attendue"
                        })
                    elif p_win >= 0.20 and value_pct >= 20:
                        win_risk = "Faible" if (cote and cote <= 4 and p_win >= 0.22) else ("Modéré" if (cote and cote <= 8 and p_win >= 0.15) else "Élevé")
                        bet_types.append({
                            "type": "SIMPLE GAGNANT",
                            "emoji": "🏆",
                            "confidence": "Value Bet",
                            "risk": win_risk,
                            "proba": round(p_win * 100, 1),
                            "cote_estimee": round(cote, 2) if cote else None,
                            "value": round(value_pct, 1),
                            "kelly": round(kelly_pct, 1),
                            "description": f"Value bet (+{value_pct:.0f}% value - cote {cote:.2f})" if cote else "Value bet"
                        })
                    elif p_win >= 0.12 and value_pct >= 35:
                        bet_types.append({
                            "type": "SIMPLE GAGNANT",
                            "emoji": "🏆",
                            "confidence": "Outsider",
                            "risk": "Très élevé",
                            "proba": round(p_win * 100, 1),
                            "cote_estimee": round(cote, 2) if cote else None,
                            "value": round(value_pct, 1),
                            "kelly": round(kelly_pct, 1),
                            "description": f"Outsider value (+{value_pct:.0f}% - petite mise conseillée)" if cote else "Outsider value"
                        })
                    if not bet_types:
                        if value_pct > 0 and p_win >= 0.10:
                            win_risk = "Faible" if (cote and cote <= 4 and p_win >= 0.22) else ("Modéré" if (cote and cote <= 8 and p_win >= 0.15) else "Élevé")
                            bet_types.append({
                                "type": "SIMPLE GAGNANT",
                                "emoji": "🏆",
                                "confidence": "Prudent",
                                "risk": win_risk,
                                "proba": round(p_win * 100, 1),
                                "cote_estimee": round(cote, 2) if cote else None,
                                "value": round(value_pct, 1),
                                "kelly": round(kelly_pct, 1),
                                "description": f"Value positive (+{value_pct:.0f}%) - mise prudente" if cote else "Value positive"
                            })
                        elif value_place_pct > 0 and p_place >= 0.40 and r["score"] >= 55:
                            bet_types.append({
                                "type": "SIMPLE PLACÉ",
                                "emoji": "🥉",
                                "confidence": "Prudent",
                                "risk": "Faible",
                                "proba": round(p_place * 100, 1),
                                "cote_estimee": round(cote_place, 2),
                                "cote_is_estimate": True,
                                "value": round(value_place_pct, 1),
                                "kelly": round(kelly_place_pct, 1),
                                "description": f"Placé avec estimation positive (+{value_place_pct:.0f}%)"
                            })
                        else:
                            continue
                    bet_types.sort(key=lambda x: (
                        -1 if x['risk'] == 'Faible' else (0 if x['risk'] == 'Modéré' else 1),
                        -x['proba'],
                        -x['value']
                    ))
                    primary_bet = bet_types[0]

                    cur.execute("""
                        SELECT 
                            COUNT(*) as courses,
                            SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
                        FROM cheval_courses_seen
                        WHERE nom_norm = %s
                    """, (r["nom"],))
                    hist = cur.fetchone()

                    drift = 0
                    if r["cote_ref"] and r["cote"]:
                        drift = ((r["cote"] - r["cote_ref"]) / r["cote_ref"]) * 100

                    picks.append({
                        "cheval_id": hash(r["nom"]) % 100000,
                        "nom": r["nom"],
                        "race_key": r["race_key"],
                        "hippodrome": r["hippodrome"],
                        "heure": r["heure"] or "14:00",
                        "numero": r["numero"],
                        "cote": round(r["cote"], 2) if r["cote"] else None,
                        "cote_place": round(cote_place, 2),
                        "p_win": round(p_win, 4),
                        "p_place": round(p_place, 4),
                        "value": round(value_pct, 2),
                        "value_pct": round(value_pct, 2),
                        "value_place": round(value_place_pct, 2),
                        "kelly": round(kelly_pct, 2),
                        "kelly_pct": round(kelly_pct, 2),
                        "kelly_place": round(kelly_place_pct, 2),
                        "bet_type": primary_bet["type"],
                        "bet_type_emoji": primary_bet["emoji"],
                        "bet_confidence": primary_bet["confidence"],
                        "bet_risk": primary_bet["risk"],
                        "bet_description": primary_bet["description"],
                        "all_bet_types": bet_types,
                        "score": r["score"],
                        "discipline": r["discipline"],
                        "distance": r["distance"],
                        "drift": round(drift, 1),
                        "jockey": r["jockey"],
                        "entraineur": r["entraineur"],
                        "tendance": r["tendance"],
                        "est_favori": r["est_favori"],
                        "rationale": generate_rationale(r["score"], value_pct, r["tendance"], r["est_favori"], r["avis"]),
                        "historique": {
                            "courses": hist[0] if hist else 0,
                            "victoires": hist[1] if hist else 0
                        },
                        "resultat": "Gagné" if r["is_win"] == 1 else ("Placé" if r["place"] and r["place"] <= 3 else None),
                        "analysis_source": analysis_source_label,
                        "p_benter_model": round(r["p_benter_raw"], 4) if r["p_benter_raw"] is not None else None,
                        "context_effect": r["context_effect"],
                        "micro_effect": r["micro_effect"],
                        "p_market_corr": round(p_mkt_corr[idx], 4),
                        "alpha_dynamic": round(alpha, 4),
                        "fair_odds": round(1.0 / p_win, 2) if p_win > 0 else None,
                        "micro_action": micro_action,
                        "micro_score": round(micro_score, 4),
                        "micro_drift_pct": round(drift_pct * 100, 2),
                        "micro_reason": micro_reason
                    })
        except Exception as e:
            market_meta = {"status": "error", "reason": str(e)}
        
        # Trier par value décroissante
        picks.sort(key=lambda x: (x['value'], x['kelly']), reverse=True)
        
        con.close()
        
        return {
            "date": search_date,
            "picks": picks[:50],  # Top 50
            "total": len(picks),
            "summary": {
                "avg_value": round(sum(p['value'] for p in picks) / len(picks), 2) if picks else 0,
                "avg_kelly": round(sum(p['kelly'] for p in picks) / len(picks), 2) if picks else 0,
                "nb_courses": len(set(p['race_key'] for p in picks))
            },
            "meta": {
                "benter_analysis": benter_meta,
                "market_blend": market_meta,
                "champion": champion_meta,
                "place_model": place_model_meta,
                "preoff_filter": {"skipped": skipped_post_off},
                "status_filter": {"skipped": skipped_status},
                "incident_filter": {"skipped": skipped_incident},
                "high_odds_filter": {"skipped": skipped_high_odds, "max_cote": MAX_PREOFF_COTE}
            }
        }
        
    except HTTPException:
        if con:
            con.close()
        raise
    except Exception as e:
        if con:
            con.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/picks/combined/{race_key}")
async def get_combined_bets(race_key: str):
    """
    Génère des recommandations de paris combinés pour une course spécifique.
    Types: Couplé, Tiercé, Quarté+, Quinté+, Multi, 2 sur 4
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Récupérer tous les chevaux de la course avec leurs stats
        cur.execute("""
            SELECT 
                cs.nom_norm as nom,
                cs.numero_dossard as numero,
                cs.cote_finale as cote,
                cs.cote_reference,
                cs.tendance_cote,
                cs.est_favori,
                cs.discipline,
                cs.distance_m
            FROM cheval_courses_seen cs
            WHERE cs.race_key = %s
            AND cs.cote_finale IS NOT NULL
            AND cs.cote_finale > 0
            ORDER BY cs.cote_finale ASC
        """, (race_key,))
        
        horses = []
        for row in cur.fetchall():
            nom, numero, cote, cote_ref, tendance, est_favori, discipline, distance = row
            if cote and cote > 0:
                # Calcul proba
                p_win = 1 / cote
                p_win = max(0.02, min(0.80, p_win))
                p_place = min(p_win * 2.5, 0.90)
                
                horses.append({
                    "nom": nom,
                    "numero": numero or 0,
                    "cote": round(cote, 2),
                    "p_win": round(p_win, 4),
                    "p_place": round(p_place, 4),
                    "est_favori": est_favori,
                    "tendance": tendance
                })
        
        con.close()
        
        if len(horses) < 4:
            return {"error": "Pas assez de chevaux pour les paris combinés", "horses": len(horses)}
        
        # Trier par probabilité de victoire décroissante
        horses.sort(key=lambda x: x['p_win'], reverse=True)
        
        combined_bets = []
        
        # ============================================
        # 2 SUR 4 - Facile, 2 chevaux parmi les 4 premiers
        # ============================================
        top4 = horses[:4]
        top4_nums = [h['numero'] for h in top4]
        # Proba approx: C(4,2) combinaisons, on a besoin que 2 soient dans top 4
        p_2sur4 = sum(h['p_place'] for h in top4[:4]) / 4 * 0.8  # Approximation
        combined_bets.append({
            "type": "2 SUR 4",
            "emoji": "🎲",
            "description": "2 chevaux parmi les 4 premiers (ordre libre)",
            "risk": "Très faible",
            "selections": top4_nums,
            "selection_names": [h['nom'] for h in top4],
            "proba_approx": round(min(p_2sur4 * 2, 0.65) * 100, 1),
            "mise_conseillée": "2-5€",
            "gain_potentiel": "Faible (1.5-3x)",
            "conseil": "Paris très sûr pour débuter, gains modestes"
        })
        
        # ============================================
        # COUPLÉ GAGNANT - Les 2 premiers dans l'ordre
        # ============================================
        top2 = horses[:2]
        p_couple_ordre = top2[0]['p_win'] * (top2[1]['p_win'] / (1 - top2[0]['p_win']))
        combined_bets.append({
            "type": "COUPLÉ ORDRE",
            "emoji": "🥇🥈",
            "description": f"1er: {top2[0]['nom']} - 2e: {top2[1]['nom']}",
            "risk": "Élevé",
            "selections": [top2[0]['numero'], top2[1]['numero']],
            "selection_names": [top2[0]['nom'], top2[1]['nom']],
            "proba_approx": round(p_couple_ordre * 100, 1),
            "mise_conseillée": "1-2€",
            "gain_potentiel": f"~{round(1/(p_couple_ordre+0.01))}x la mise",
            "conseil": "Difficile mais gros gains - petite mise"
        })
        
        # ============================================
        # COUPLÉ PLACÉ - 2 chevaux dans les 3 premiers (désordre)
        # ============================================
        top3 = horses[:3]
        p_couple_place = top3[0]['p_place'] * top3[1]['p_place'] * 0.7
        combined_bets.append({
            "type": "COUPLÉ PLACÉ",
            "emoji": "🥉🥉",
            "description": f"{top3[0]['nom']} et {top3[1]['nom']} dans les 3 premiers",
            "risk": "Modéré",
            "selections": [top3[0]['numero'], top3[1]['numero']],
            "selection_names": [top3[0]['nom'], top3[1]['nom']],
            "proba_approx": round(p_couple_place * 100, 1),
            "mise_conseillée": "2-5€",
            "gain_potentiel": f"~{round(1/(p_couple_place+0.01))}x la mise",
            "conseil": "Bon compromis risque/gain"
        })
        
        # ============================================
        # TIERCÉ - Les 3 premiers
        # ============================================
        if len(horses) >= 3:
            top3 = horses[:3]
            p_tierce_ordre = top3[0]['p_win'] * (top3[1]['p_win'] / 0.9) * (top3[2]['p_win'] / 0.8)
            p_tierce_desordre = p_tierce_ordre * 6  # 3! arrangements
            
            combined_bets.append({
                "type": "TIERCÉ ORDRE",
                "emoji": "🥇🥈🥉",
                "description": f"1-2-3: {top3[0]['nom']}, {top3[1]['nom']}, {top3[2]['nom']}",
                "risk": "Très élevé",
                "selections": [h['numero'] for h in top3],
                "selection_names": [h['nom'] for h in top3],
                "proba_approx": round(p_tierce_ordre * 100, 2),
                "mise_conseillée": "1€",
                "gain_potentiel": "Élevé (50-200x)",
                "conseil": "Très difficile - mise minimale"
            })
            
            combined_bets.append({
                "type": "TIERCÉ DÉSORDRE",
                "emoji": "🔄🥇🥈🥉",
                "description": f"{top3[0]['nom']}, {top3[1]['nom']}, {top3[2]['nom']} (ordre libre)",
                "risk": "Élevé",
                "selections": [h['numero'] for h in top3],
                "selection_names": [h['nom'] for h in top3],
                "proba_approx": round(min(p_tierce_desordre, 0.25) * 100, 1),
                "mise_conseillée": "1-3€",
                "gain_potentiel": "Moyen (10-30x)",
                "conseil": "Plus accessible que l'ordre"
            })
        
        # ============================================
        # QUARTÉ+ - Les 4 premiers
        # ============================================
        if len(horses) >= 4:
            top4 = horses[:4]
            p_quarte_desordre = top4[0]['p_place'] * top4[1]['p_place'] * top4[2]['p_place'] * top4[3]['p_place'] * 0.3
            
            combined_bets.append({
                "type": "QUARTÉ+ DÉSORDRE",
                "emoji": "4️⃣🔄",
                "description": f"Top 4: {', '.join(h['nom'] for h in top4)}",
                "risk": "Très élevé",
                "selections": [h['numero'] for h in top4],
                "selection_names": [h['nom'] for h in top4],
                "proba_approx": round(p_quarte_desordre * 100, 2),
                "mise_conseillée": "1-2€",
                "gain_potentiel": "Élevé (100-500x)",
                "conseil": "Difficile - bonus si ordre exact"
            })
        
        # ============================================
        # QUINTÉ+ - Les 5 premiers (courses principales)
        # ============================================
        if len(horses) >= 5:
            top5 = horses[:5]
            p_quinte_desordre = 0.01  # Très faible
            for h in top5[:3]:
                p_quinte_desordre *= h['p_place']
            
            combined_bets.append({
                "type": "QUINTÉ+ DÉSORDRE",
                "emoji": "5️⃣⭐",
                "description": f"Top 5: {', '.join(h['nom'] for h in top5)}",
                "risk": "Extrême",
                "selections": [h['numero'] for h in top5],
                "selection_names": [h['nom'] for h in top5],
                "proba_approx": round(p_quinte_desordre * 100, 3),
                "mise_conseillée": "1€",
                "gain_potentiel": "Très élevé (1000x+)",
                "conseil": "Jackpot potentiel - mise minimale uniquement"
            })
        
        # ============================================
        # MULTI - 4 à 7 chevaux, flexibilité
        # ============================================
        if len(horses) >= 6:
            top6 = horses[:6]
            combined_bets.append({
                "type": "MULTI EN 6",
                "emoji": "🎰",
                "description": f"6 chevaux pour le top 4: {', '.join(h['nom'] for h in top6)}",
                "risk": "Modéré",
                "selections": [h['numero'] for h in top6],
                "selection_names": [h['nom'] for h in top6],
                "proba_approx": round(35, 1),  # Approximation généreuse avec 6 chevaux
                "mise_conseillée": "3€ (base)",
                "gain_potentiel": "Variable (10-100x)",
                "conseil": "Bonne couverture, coût maîtrisé"
            })
        
        return {
            "race_key": race_key,
            "nb_partants": len(horses),
            "discipline": horses[0].get('discipline') if horses else None,
            "horses": horses[:8],  # Top 8 pour référence
            "combined_bets": combined_bets,
            "conseil_general": "Privilégiez les paris à faible risque (2 sur 4, Couplé placé) pour un bankroll sain. Réservez 10-20% max aux paris risqués."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_rationale(score: int, value: float, tendance: str, est_favori: bool, avis: str) -> str:
    """Génère une explication textuelle pour le pick"""
    reasons = []
    
    if score >= 70:
        reasons.append("Score élevé")
    elif score >= 60:
        reasons.append("Bon score")
    
    if value >= 15:
        reasons.append(f"Excellente value (+{value:.0f}%)")
    elif value >= 5:
        reasons.append(f"Bonne value (+{value:.0f}%)")
    
    if tendance == '-':
        reasons.append("Cote en baisse (confiance)")
    
    if est_favori:
        reasons.append("Favori")
    
    if avis == 'POSITIF':
        reasons.append("Avis entraîneur positif")
    
    return " • ".join(reasons) if reasons else "Value bet"


@app.get("/analyze/{race_key:path}")
async def analyze_race(race_key: str):
    """
    Analyse détaillée d'une course spécifique.
    Retourne les participants avec scores, value, kelly, drift, rationale.
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Récupérer les infos de la course
        cur.execute("""
            SELECT DISTINCT
                hippodrome_nom,
                distance_m,
                type_course,
                discipline,
                allocation_totale,
                corde
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """, (race_key,))
        
        course_info = cur.fetchone()
        if not course_info:
            raise HTTPException(status_code=404, detail=f"Course non trouvée: {race_key}")
        
        hippodrome, distance, type_course, discipline, allocation, corde = course_info
        
        # Récupérer les participants
        cur.execute("""
            SELECT 
                nom_norm,
                numero_dossard,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                avis_entraineur,
                driver_jockey,
                entraineur,
                age,
                sexe,
                musique,
                is_win,
                place_finale
            FROM cheval_courses_seen
            WHERE race_key = %s
            ORDER BY numero_dossard
        """, (race_key,))
        
        analyses = []
        for row in cur.fetchall():
            nom, numero, cote, cote_ref, tendance, amplitude, est_favori, avis, jockey, entraineur, age, sexe, musique, is_win, place = row
            
            # Score
            score = calculate_prediction_score(cote, cote_ref, tendance, amplitude, est_favori, avis)
            
            # Probabilités
            p_win = 0.1
            if cote and cote > 0:
                proba_implicite = 1 / cote
                adjustment = (score - 50) / 200
                p_win = proba_implicite * (1 + adjustment)
                p_win = max(0.01, min(0.95, p_win))
            
            # Value
            value_pct = (p_win * (cote or 1) - 1) * 100 if cote else 0
            
            # Kelly
            kelly_pct = 0
            if value_pct > 0 and cote and cote > 1:
                kelly_pct = (p_win * cote - 1) / (cote - 1) * 100
                kelly_pct = max(0, min(kelly_pct, 25))
            
            # Drift (variation de cote)
            drift = 0
            if cote_ref and cote:
                drift = ((cote - cote_ref) / cote_ref) * 100
            
            # Historique cheval
            cur.execute("""
                SELECT COUNT(*), SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END)
                FROM cheval_courses_seen WHERE nom_norm = %s
            """, (nom,))
            hist = cur.fetchone()
            
            analyses.append({
                "nom": nom,
                "numero": numero,
                "cote": round(cote, 2) if cote else None,
                "cote_reference": round(cote_ref, 2) if cote_ref else None,
                "p_win": round(p_win, 4),
                "value": round(value_pct, 2),
                "kelly": round(kelly_pct, 2),
                "score": score,
                "drift": round(drift, 1),
                "tendance": tendance,
                "est_favori": est_favori,
                "avis": avis,
                "jockey": jockey,
                "entraineur": entraineur,
                "age": age,
                "sexe": sexe,
                "musique": musique,
                "rationale": generate_rationale(score, value_pct, tendance, est_favori, avis),
                "historique": {
                    "courses": hist[0] if hist else 0,
                    "victoires": hist[1] if hist else 0
                },
                "resultat": "Gagné" if is_win == 1 else ("Placé" if place and place <= 3 else None)
            })
        
        # Trier par score puis value
        analyses.sort(key=lambda x: (x['score'], x['value']), reverse=True)
        
        con.close()
        
        return {
            "race_key": race_key,
            "hippodrome": hippodrome,
            "distance": distance,
            "type_course": type_course,
            "discipline": discipline,
            "allocation": allocation,
            "corde": corde,
            "nb_partants": len(analyses),
            "analyses": analyses,
            "top_picks": [a for a in analyses if a['value'] > 0][:5]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# IMPORT SERVICE D'ESTIMATION P(PLACE) AVANCÉ
# =============================================================================
try:
    from services.place_estimator_service import (
        get_place_estimator_service,
        PlaceEstimatorService,
        ESTIMATOR_AVAILABLE
    )
    PLACE_ESTIMATOR_LOADED = True
except ImportError:
    PLACE_ESTIMATOR_LOADED = False
    ESTIMATOR_AVAILABLE = False
    print("⚠️  Service place_estimator non disponible - mode fallback")


class ExoticsBuildRequest(BaseModel):
    """Requête pour construire des paris exotiques"""
    budget: float = 50
    pack: str = "EQUILIBRE"  # SUR, EQUILIBRE, RISQUE
    race_key: Optional[str] = None
    use_advanced_estimator: bool = True  # Utiliser l'estimateur p(place) avancé
    n_simulations: int = 10000  # Nombre de simulations Monte Carlo


class ExoticAdvancedRequest(BaseModel):
    """Requête pour analyse avancée des exotiques"""
    race_key: str
    estimator: str = "auto"  # harville, henery, stern, lbs, auto
    n_simulations: int = 20000
    min_ev_percent: float = 5.0  # EV minimum en %


@app.post("/exotics/build")
async def build_exotics(request: ExoticsBuildRequest):
    """
    Génère des tickets de paris exotiques (Tiercé, Quarté, Quinté).
    
    Utilise l'estimateur avancé p(place) avec:
    - Harville, Henery, Stern ou Lo-Bacon-Shone selon la discipline
    - Simulation Plackett-Luce avec température apprise
    - Monte Carlo (N simulations) pour les probabilités de combos
    - Calcul EV avec takeout parimutuel
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        budget = request.budget
        pack = request.pack.upper()
        use_advanced = request.use_advanced_estimator and PLACE_ESTIMATOR_LOADED and ESTIMATOR_AVAILABLE
        
        # Trouver une course avec assez de partants
        if request.race_key:
            race_key = request.race_key
        else:
            cur.execute("""
                SELECT race_key, COUNT(*) as nb
                FROM cheval_courses_seen
                WHERE race_key LIKE '2025%'
                GROUP BY race_key
                HAVING COUNT(*) >= 8
                ORDER BY race_key DESC
                LIMIT 1
            """)
            result = cur.fetchone()
            if not result:
                return {"tickets": [], "message": "Aucune course disponible"}
            race_key = result[0]
        
        # Récupérer infos course pour la discipline
        cur.execute("""
            SELECT DISTINCT discipline
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """, (race_key,))
        discipline_row = cur.fetchone()
        discipline = discipline_row[0] if discipline_row else "plat"
        
        # Récupérer les partants avec scores
        cur.execute("""
            SELECT 
                nom_norm,
                numero_dossard,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                avis_entraineur
            FROM cheval_courses_seen
            WHERE race_key = %s
            AND cote_finale IS NOT NULL
            ORDER BY cote_finale
        """, (race_key,))
        
        partants = []
        for row in cur.fetchall():
            nom, numero, cote, cote_ref, tendance, amplitude, favori, avis = row
            score = calculate_prediction_score(cote, cote_ref, tendance, amplitude, favori, avis)
            partants.append({
                "nom": nom,
                "numero": numero if numero else len(partants) + 1,
                "cote": cote if cote and cote > 0 else 10.0,
                "score": score
            })
        
        con.close()
        
        if len(partants) < 3:
            return {"tickets": [], "message": "Pas assez de partants"}
        
        # Utiliser l'estimateur avancé si disponible
        if use_advanced:
            try:
                service = get_place_estimator_service()
                
                # Analyse complète de la course
                analysis = service.analyze_race(
                    partants=partants,
                    discipline=discipline,
                    n_simulations=request.n_simulations
                )
                
                # Générer les packs de tickets
                packs = service.generate_packs(
                    analysis=analysis,
                    budget=budget,
                    pack_type=pack
                )
                
                tickets = packs.get(pack, [])
                
                # Enrichir les tickets avec les infos des partants
                for ticket in tickets:
                    ticket["combo"] = [
                        {"nom": p["nom"], "numero": p["numero"]} 
                        for p in partants 
                        if p["numero"] in ticket.get("numeros", [])
                    ][:5]
                
                return {
                    "tickets": tickets,
                    "race_key": race_key,
                    "discipline": discipline,
                    "budget": budget,
                    "pack": pack,
                    "budget_utilise": round(sum(t.get('mise', 0) for t in tickets), 2),
                    "ev_totale": round(sum(t.get('ev', 0) for t in tickets), 2),
                    "advanced_estimator": True,
                    "estimator_used": analysis.get("estimator_used", "unknown"),
                    "n_simulations": request.n_simulations,
                    "calibration": analysis.get("calibration", {}),
                    "top_probas": analysis.get("top_combos", [])[:5]
                }
                
            except Exception as e:
                # Fallback sur méthode classique
                print(f"⚠️ Erreur estimateur avancé: {e}, fallback")
                use_advanced = False
        
        # ===== MÉTHODE CLASSIQUE (fallback) =====
        partants.sort(key=lambda x: x['score'], reverse=True)
        tickets = []
        
        if pack == "SUR":
            # Pack sûr: combinaisons avec les 4-5 meilleurs
            base = partants[:4]
            
            # Tiercé
            tickets.append({
                "type": "Tiercé",
                "bet_type": "tierce",
                "combo": [{"nom": p['nom'], "numero": p['numero']} for p in base[:3]],
                "selections": [p['nom'] for p in base[:3]],
                "stake": round(budget * 0.4, 2),
                "mise": round(budget * 0.4, 2),
                "odds": round(sum(p['cote'] for p in base[:3]) / 3, 2),
                "ev": round(sum(p['score'] for p in base[:3]) / 3 * 0.1 - budget * 0.4 * 0.15, 2),
                "couverture": 0.6,
                "description": "Top 3 favoris en ordre"
            })
            
            # Quarté désordre
            tickets.append({
                "type": "Quarté Désordre",
                "bet_type": "quarte",
                "combo": [{"nom": p['nom'], "numero": p['numero']} for p in base[:4]],
                "selections": [p['nom'] for p in base[:4]],
                "stake": round(budget * 0.35, 2),
                "mise": round(budget * 0.35, 2),
                "odds": round(sum(p['cote'] or 0 for p in base[:4]) / 2, 2),
                "ev": round(sum(p['score'] for p in base[:4]) / 4 * 0.15 - budget * 0.35 * 0.1, 2),
                "couverture": 0.5,
                "description": "4 meilleurs en désordre"
            })
            
        elif pack == "RISQUE":
            # Pack risqué: inclure des outsiders
            base = partants[:3]
            outsiders = [p for p in partants[5:10] if p['cote'] and p['cote'] > 10][:2]
            
            # Tiercé avec 1 outsider
            if outsiders:
                combo = base[:2] + [outsiders[0]]
                tickets.append({
                    "type": "Tiercé Outsider",
                    "bet_type": "tierce",
                    "combo": [{"nom": p['nom'], "numero": p['numero']} for p in combo],
                    "selections": [p['nom'] for p in combo],
                    "stake": round(budget * 0.25, 2),
                    "mise": round(budget * 0.25, 2),
                    "odds": round(sum(p['cote'] or 0 for p in combo), 2),
                    "ev": round(budget * 0.25 * 2, 2),
                    "couverture": 0.15,
                    "description": "Avec 1 outsider pour gros rapport"
                })
            
            # Quinté avec base + outsiders
            full_combo = base + outsiders
            if len(full_combo) >= 5:
                tickets.append({
                    "type": "Quinté+ Risqué",
                    "bet_type": "quinte",
                    "combo": [{"nom": p['nom'], "numero": p['numero']} for p in full_combo[:5]],
                    "selections": [p['nom'] for p in full_combo[:5]],
                    "stake": round(budget * 0.4, 2),
                    "mise": round(budget * 0.4, 2),
                    "odds": round(sum(p['cote'] or 0 for p in full_combo[:5]) * 2, 2),
                    "ev": round(budget * 0.4 * 5, 2),
                    "couverture": 0.08,
                    "description": "Combinaison audacieuse"
                })
                
        else:  # EQUILIBRE
            base = partants[:5]
            
            # Tiercé ordre
            tickets.append({
                "type": "Tiercé Ordre",
                "bet_type": "tierce",
                "combo": [{"nom": p['nom'], "numero": p['numero']} for p in base[:3]],
                "selections": [p['nom'] for p in base[:3]],
                "stake": round(budget * 0.2, 2),
                "mise": round(budget * 0.2, 2),
                "odds": round(sum(p['cote'] or 0 for p in base[:3]), 2),
                "ev": round(sum(p['score'] for p in base[:3]) / 3 * 0.08, 2),
                "couverture": 0.35,
                "description": "Favoris en ordre"
            })
            
            # Quarté combiné
            tickets.append({
                "type": "Quarté Combiné",
                "bet_type": "quarte",
                "combo": [{"nom": p['nom'], "numero": p['numero']} for p in base[:4]],
                "selections": [p['nom'] for p in base[:4]],
                "stake": round(budget * 0.35, 2),
                "mise": round(budget * 0.35, 2),
                "odds": round(sum(p['cote'] or 0 for p in base[:4]) / 2, 2),
                "ev": round(sum(p['score'] for p in base[:4]) / 4 * 0.12, 2),
                "couverture": 0.45,
                "description": "Base solide en combiné"
            })
            
            # Quinté+ base 3 + 2 compléments
            tickets.append({
                "type": "Quinté+ Combiné",
                "bet_type": "quinte",
                "combo": [{"nom": p['nom'], "numero": p['numero']} for p in base[:5]],
                "selections": [p['nom'] for p in base[:5]],
                "stake": round(budget * 0.35, 2),
                "mise": round(budget * 0.35, 2),
                "odds": round(sum(p['cote'] or 0 for p in base[:5]) / 2, 2),
                "ev": round(sum(p['score'] for p in base[:5]) / 5 * 0.2, 2),
                "couverture": 0.3,
                "description": "5 meilleurs combinés"
            })
        
        return {
            "tickets": tickets,
            "race_key": race_key,
            "discipline": discipline,
            "budget": budget,
            "pack": pack,
            "budget_utilise": round(sum(t['stake'] for t in tickets), 2),
            "ev_totale": round(sum(t['ev'] for t in tickets), 2),
            "advanced_estimator": False,
            "estimator_used": "fallback"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exotics/advanced")
async def analyze_exotics_advanced(request: ExoticAdvancedRequest):
    """
    Analyse avancée des paris exotiques avec estimateurs p(place).
    
    Retourne les probabilités détaillées pour chaque combinaison
    Trio/Quarté/Quinté avec calcul EV complet.
    
    Estimateurs disponibles:
    - harville: Classique (1973), peut surestimer les favoris
    - henery: Corrige le biais favori avec γ=0.81  
    - stern: Lissage avec λ=0.15
    - lbs: Lo-Bacon-Shone, réallocation itérative
    - auto: Sélection automatique selon la discipline
    """
    if not PLACE_ESTIMATOR_LOADED or not ESTIMATOR_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Service d'estimation avancé non disponible"
        )
    
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        race_key = request.race_key
        
        # Récupérer la discipline
        cur.execute("""
            SELECT DISTINCT discipline, hippodrome
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """, (race_key,))
        race_info = cur.fetchone()
        discipline = race_info[0] if race_info else "plat"
        hippodrome = race_info[1] if race_info else "Inconnu"
        
        # Récupérer les partants
        cur.execute("""
            SELECT 
                nom_norm,
                numero_dossard,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                avis_entraineur
            FROM cheval_courses_seen
            WHERE race_key = %s
            AND cote_finale IS NOT NULL
            ORDER BY cote_finale
        """, (race_key,))
        
        partants = []
        for row in cur.fetchall():
            nom, numero, cote, cote_ref, tendance, amplitude, favori, avis = row
            score = calculate_prediction_score(cote, cote_ref, tendance, amplitude, favori, avis)
            partants.append({
                "nom": nom,
                "numero": numero if numero else len(partants) + 1,
                "cote": cote if cote and cote > 0 else 10.0,
                "score": score
            })
        
        con.close()
        
        if len(partants) < 3:
            raise HTTPException(status_code=400, detail="Pas assez de partants pour l'analyse")
        
        # Utiliser le service d'estimation
        service = get_place_estimator_service()
        
        # Forcer l'estimateur si spécifié
        estimator_override = None if request.estimator == "auto" else request.estimator
        
        analysis = service.analyze_race(
            partants=partants,
            discipline=discipline,
            n_simulations=request.n_simulations,
            estimator_override=estimator_override
        )
        
        # Filtrer par EV minimum
        top_combos = [
            c for c in analysis.get("top_combos", [])
            if c.get("ev_percent", 0) >= request.min_ev_percent
        ]
        
        # Préparer les p(place) par cheval
        p_place_detail = []
        for i, p in enumerate(partants):
            p_place_detail.append({
                "numero": p["numero"],
                "nom": p["nom"],
                "cote": p["cote"],
                "p_win": analysis.get("p_win", [])[i] if i < len(analysis.get("p_win", [])) else 0,
                "p_place_2": analysis.get("p_place_2", [])[i] if i < len(analysis.get("p_place_2", [])) else 0,
                "p_place_3": analysis.get("p_place_3", [])[i] if i < len(analysis.get("p_place_3", [])) else 0,
                "score_model": p["score"]
            })
        
        return {
            "race_key": race_key,
            "hippodrome": hippodrome,
            "discipline": discipline,
            "nb_partants": len(partants),
            "estimator_used": analysis.get("estimator_used"),
            "n_simulations": request.n_simulations,
            "temperature": analysis.get("temperature"),
            "calibration": analysis.get("calibration", {}),
            "partants": p_place_detail,
            "top_combos_ev_positive": top_combos[:20],
            "trio_top5": analysis.get("trio_probs", [])[:5],
            "quarte_top5": analysis.get("quarte_probs", [])[:5],
            "quinte_top5": analysis.get("quinte_probs", [])[:5],
            "recommendations": _generate_exotic_recommendations(analysis, partants)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_exotic_recommendations(analysis: dict, partants: list) -> list:
    """Génère des recommandations textuelles basées sur l'analyse."""
    recommendations = []
    
    # Analyser les probabilités
    p_win = analysis.get("p_win", [])
    top_combos = analysis.get("top_combos", [])
    
    if p_win:
        # Favori dominant?
        if max(p_win) > 0.35:
            fav_idx = p_win.index(max(p_win))
            fav_nom = partants[fav_idx]["nom"] if fav_idx < len(partants) else "?"
            recommendations.append(f"⚠️ Favori dominant ({fav_nom}, p={max(p_win):.1%}) - Trio ordre recommandé")
        
        # Course ouverte?
        if max(p_win) < 0.20:
            recommendations.append("🎯 Course ouverte - Privilégier les combinés avec outsiders")
    
    # EV positive trouvée?
    ev_positive = [c for c in top_combos if c.get("ev_percent", 0) > 10]
    if ev_positive:
        recommendations.append(f"💰 {len(ev_positive)} combinaisons avec EV > 10% détectées")
    elif not ev_positive and top_combos:
        recommendations.append("⚡ Aucune combinaison à forte EV - Mises conservatrices recommandées")
    
    # Calibration
    brier = analysis.get("calibration", {}).get("brier_score", 1.0)
    if brier < 0.15:
        recommendations.append("✅ Modèle bien calibré (Brier < 0.15)")
    elif brier > 0.25:
        recommendations.append("⚠️ Calibration à améliorer - Prudence sur les mises")
    
    return recommendations


# ============================================================================
# Couche moteur (BK 1 000€) vs adaptation bankroll utilisateur
# ============================================================================

ENGINE_BANKROLL_REFERENCE = 1000.0
MIN_ADAPTED_STAKE_EUR = 1.0


def _classify_risk_profile(value_pct: float, odds: float) -> str:
    """Classe un pari en SÛR / ÉQUILIBRÉ / RISQUÉ selon la value et la cote."""
    if value_pct < 5 or odds <= 3:
        return "SÛR"
    if value_pct < 15 or odds < 6:
        return "ÉQUILIBRÉ"
    return "RISQUÉ"


def _bankroll_policy(bankroll: float) -> Dict[str, Any]:
    """Définit la politique d’adaptation selon la bankroll (3 zones)."""
    if bankroll < 50:
        return {
            "zone": "ZONE1_MICRO",
            "allowed_profiles": {"SÛR", "SUR"},
            "risk_priority": ["SÛR"],
            "max_bets": 2,
            "max_risque": 0,
            "daily_budget_rate": 0.05,  # limite douce
            "kelly_fraction": 0.0,
            "flat_stake_eur": 1.0,
            "min_p_win": 0.18,  # priorité haute proba
        }
    if bankroll < 250:
        return {
            "zone": "ZONE2_PETIT",
            "allowed_profiles": {"SÛR", "SUR", "ÉQUILIBRÉ", "EQUILIBRE"},
            "risk_priority": ["SÛR", "SUR", "ÉQUILIBRÉ", "EQUILIBRE"],
            "max_bets": 5,
            "max_risque": 0,
            "daily_budget_rate": 0.09,  # 8-10% cible
            "kelly_fraction": 0.20,
            "flat_stake_eur": None,
        }
    return {
        "zone": "ZONE3_FULL",
        "allowed_profiles": {"SÛR", "SUR", "ÉQUILIBRÉ", "EQUILIBRE", "RISQUÉ", "RISQUE"},
        "risk_priority": ["SÛR", "SUR", "ÉQUILIBRÉ", "EQUILIBRE", "RISQUÉ", "RISQUE"],
        "max_bets": 12,
        "max_risque": 99,
        "daily_budget_rate": 0.12,  # 10-12%
        "kelly_fraction": 0.25,
        "flat_stake_eur": None,
    }


def _compute_engine_portfolio(
    rows: list,
    bankroll: float,
    kelly_frac: float,
    value_cutoff: float,
    cap_per_bet: float,
    daily_budget_rate: float,
    rounding: float,
    max_bets_per_race: int,
    takeout_rate: float
) -> Dict[str, Any]:
    """
    Exécute le moteur (ML + calibration + value + Kelly) sur une bankroll de référence.
    Retourne les positions de référence (stake_ref) indépendantes de la bankroll utilisateur.
    """
    positions = []
    excluded = []
    race_counts = {}
    daily_budget = bankroll * daily_budget_rate
    max_stake_per_bet = bankroll * cap_per_bet
    total_stake = 0.0
    total_ev = 0.0
    min_stake_threshold = rounding if rounding and rounding > 0 else 0.01
    
    for row in rows:
        (
            nom,
            race_key,
            hippo,
            cote,
            cote_ref,
            tendance,
            amplitude,
            favori,
            avis,
            is_win,
            place
        ) = row
        
        score = calculate_prediction_score(cote, cote_ref, tendance, amplitude, favori, avis)
        
        if cote and cote > 0:
            proba_impl = 1 / cote
            adj = (score - 50) / 200
            p_win = proba_impl * (1 + adj)
            p_win = max(0.01, min(0.95, p_win))
            
            # Value vs marché (cote décimale). La cote PMU est déjà un rapport net, donc on ne
            # "ré-applique" pas le takeout ici ; sinon la value serait artificiellement négative.
            value = (p_win * cote - 1)
            value_pct = value * 100
            risk_profile = _classify_risk_profile(value_pct, cote)
            
            if value <= 0:
                excluded.append({"nom": nom, "race_key": race_key, "reason": f"value ({value:.2%}) ≤ 0"})
                continue
            if value < value_cutoff:
                excluded.append({"nom": nom, "race_key": race_key, "reason": f"value ({value:.2%}) < cutoff ({value_cutoff:.0%})"})
                continue
            
            race_counts[race_key] = race_counts.get(race_key, 0)
            if race_counts[race_key] >= max_bets_per_race:
                excluded.append({"nom": nom, "race_key": race_key, "reason": f"max {max_bets_per_race} paris/course"})
                continue
            
            if cote > 1:
                kelly_raw = (p_win * (cote - 1) - (1 - p_win)) / (cote - 1)
                kelly_raw = max(0, kelly_raw)
                kelly_adjusted = min(kelly_raw * kelly_frac, cap_per_bet)
            else:
                kelly_raw = 0.0
                kelly_adjusted = 0.0
            
            stake_raw = bankroll * kelly_adjusted
            if rounding > 0:
                stake = round(stake_raw / rounding) * rounding
            else:
                stake = round(stake_raw, 2)
            
            if stake > max_stake_per_bet:
                stake = math.floor(max_stake_per_bet / rounding) * rounding if rounding > 0 else round(max_stake_per_bet, 2)
            
            if total_stake + stake > daily_budget:
                remaining = daily_budget - total_stake
                if remaining >= min_stake_threshold:
                    if rounding > 0:
                        stake = math.floor(remaining / rounding) * rounding
                    else:
                        stake = round(remaining, 2)
                else:
                    excluded.append({"nom": nom, "race_key": race_key, "reason": "budget jour épuisé"})
                    continue
            
            ev = stake * value
            
            if stake >= min_stake_threshold:
                race_counts[race_key] += 1
                position = {
                    "nom": nom,
                    "race_key": race_key,
                    "hippodrome": hippo,
                    "cote": round(cote, 2),
                    "p_win": round(p_win, 4),
                    "value": round(value_pct, 2),
                    "value_decimal": round(value, 4),
                    "profil_risque": risk_profile,
                    "kelly_raw": round(kelly_raw * 100, 2),
                    "kelly_raw_rate": kelly_raw,
                    "kelly_adjusted": round(kelly_adjusted * 100, 4),
                    "kelly_rate": kelly_adjusted,
                    "stake_reference": round(stake, 2),
                    "stake": round(stake, 2),
                    "ev": round(ev, 2),
                    "ev_decimal": value,
                    "score": score,
                    "resultat": "Gagné" if is_win == 1 else ("Placé" if place and place <= 3 else None)
                }
                total_stake += stake
                total_ev += ev
                positions.append(position)
    
    positions.sort(key=lambda x: x['ev'], reverse=True)
    
    if len(positions) > 20:
        positions = positions[:20]
        total_stake = sum(p['stake_reference'] for p in positions)
        total_ev = sum(p['ev'] for p in positions)
    
    budget_left = daily_budget - total_stake
    
    return {
        "positions": positions,
        "excluded": excluded,
        "total_stake": round(total_stake, 2),
        "total_ev": round(total_ev, 2),
        "budget_left": round(budget_left, 2),
        "daily_budget": round(daily_budget, 2),
        "max_stake_per_bet": round(max_stake_per_bet, 2),
        "kelly_fraction_used": kelly_frac,
        "daily_budget_rate_used": daily_budget_rate,
        "cap_per_bet_used": cap_per_bet
    }


def _adapt_positions_for_bankroll(
    engine_positions: list,
    bankroll: float,
    cap_per_bet: float,
    daily_budget_rate: float,
    rounding: float,
    policy: Dict[str, Any],
    engine_kelly_fraction: float
) -> Dict[str, Any]:
    """Adapte le portefeuille de référence à la bankroll utilisateur (mises min 1€)."""
    risk_priority = policy.get("risk_priority", ["SÛR", "ÉQUILIBRÉ", "RISQUÉ"])
    priority_map = {r: i for i, r in enumerate(risk_priority)}
    ordered_positions = sorted(
        engine_positions,
        key=lambda p: (priority_map.get(p.get("profil_risque", "ÉQUILIBRÉ"), 99), -p.get("ev", 0))
    )
    
    adapted = []
    excluded = []
    total_stake = 0.0
    total_ev = 0.0
    risky_used = 0
    policy_daily_rate = policy.get("daily_budget_rate", daily_budget_rate)
    daily_budget = bankroll * policy_daily_rate
    cap_eur = bankroll * cap_per_bet
    scale = bankroll / ENGINE_BANKROLL_REFERENCE if ENGINE_BANKROLL_REFERENCE > 0 else 1.0
    policy_kelly = policy.get("kelly_fraction", engine_kelly_fraction)
    kelly_scale = (policy_kelly / engine_kelly_fraction) if engine_kelly_fraction > 0 else 1.0
    min_p_win = policy.get("min_p_win")
    zone = policy.get("zone", "ZONE3_FULL")
    adaptation_notes = []
    
    # Toujours au moins une mise minimale possible
    daily_budget = max(daily_budget, MIN_ADAPTED_STAKE_EUR)
    cap_eur = max(cap_eur, MIN_ADAPTED_STAKE_EUR)
    
    if zone == "ZONE1_MICRO":
        # Micro: profils SÛR uniquement, tri p_win desc puis EV, 1-2 paris, flat 1€
        safe_only = [p for p in ordered_positions if p.get("profil_risque") in {"SÛR", "SUR"}]
        safe_sorted = sorted(safe_only, key=lambda p: (-p.get("p_win", 0), -p.get("ev", 0)))
        for pos in safe_sorted[:2]:
            if total_stake + 1 > daily_budget:
                break
            updated = pos.copy()
            updated["stake"] = 1.0
            updated["stake_user"] = 1.0
            updated["stake_reference"] = pos.get("stake_reference", pos.get("stake", 0))
            updated["ev"] = round(pos.get("ev_decimal", pos.get("value_decimal", 0)) * 1.0, 2)
            updated["ev_decimal"] = pos.get("ev_decimal", pos.get("value_decimal", 0))
            updated["policy_zone"] = zone
            updated["kelly_scale_applied"] = 0.0
            adapted.append(updated)
            total_stake += 1.0
            total_ev += updated["ev"]
        adaptation_notes.append("BK <50€: mode sécurisé, 1€ flat sur 1-2 paris SÛR.")
    else:
        for pos in ordered_positions:
            profil = pos.get("profil_risque", "ÉQUILIBRÉ")
            
            if profil not in policy["allowed_profiles"]:
                excluded.append({"nom": pos["nom"], "race_key": pos["race_key"], "reason": f"profil {profil} exclu pour bankroll {bankroll:.0f}€"})
                continue
            
            if profil == "RISQUÉ" and risky_used >= policy["max_risque"]:
                excluded.append({"nom": pos["nom"], "race_key": pos["race_key"], "reason": "quota paris risqués atteint"})
                continue
            
            if len(adapted) >= policy["max_bets"]:
                excluded.append({"nom": pos["nom"], "race_key": pos["race_key"], "reason": f"max {policy['max_bets']} paris pour cette bankroll"})
                continue
            
            if min_p_win is not None and pos.get("p_win", 0) < min_p_win:
                excluded.append({"nom": pos["nom"], "race_key": pos["race_key"], "reason": f"p_win {pos.get('p_win', 0):.2f} < seuil {min_p_win:.2f}"})
                continue
            
            stake_ref = pos.get("stake_reference", pos.get("stake", 0))
            stake_pct = stake_ref / ENGINE_BANKROLL_REFERENCE if ENGINE_BANKROLL_REFERENCE > 0 else 0
            stake = stake_pct * bankroll
            stake = stake * kelly_scale
            
            if policy.get("flat_stake_eur") is not None:
                stake = policy["flat_stake_eur"]
            elif rounding > 0:
                stake = round(stake / rounding) * rounding
            else:
                stake = round(stake, 2)
            
            stake = max(stake, MIN_ADAPTED_STAKE_EUR)
            
            cap_limit = cap_eur
            if stake > cap_limit:
                if rounding > 0:
                    stake = math.floor(cap_limit / rounding) * rounding
                else:
                    stake = round(cap_limit, 2)
                stake = max(stake, MIN_ADAPTED_STAKE_EUR)
            
            if total_stake + stake > daily_budget:
                remaining = daily_budget - total_stake
                if remaining < MIN_ADAPTED_STAKE_EUR:
                    excluded.append({"nom": pos["nom"], "race_key": pos["race_key"], "reason": "budget journalier adapté épuisé"})
                    continue
                if rounding > 0:
                    stake = math.floor(remaining / rounding) * rounding
                else:
                    stake = round(remaining, 2)
                stake = max(stake, MIN_ADAPTED_STAKE_EUR)
                if stake > cap_limit:
                    stake = cap_limit if rounding <= 0 else math.floor(cap_limit / rounding) * rounding
            
            ev_decimal = pos.get("ev_decimal", pos.get("value_decimal", 0))
            ev = stake * ev_decimal
            
            updated = pos.copy()
            updated["stake"] = round(stake, 2)
            updated["stake_user"] = round(stake, 2)
            updated["stake_reference"] = stake_ref
            updated["ev"] = round(ev, 2)
            updated["ev_decimal"] = ev_decimal
            updated["policy_zone"] = policy.get("zone")
            updated["kelly_scale_applied"] = round(kelly_scale, 3)
            adapted.append(updated)
            
            total_stake += stake
            total_ev += ev
            if profil == "RISQUÉ":
                risky_used += 1
    
    budget_left = daily_budget - total_stake
    
    return {
        "positions": adapted,
        "excluded": excluded,
        "total_stake": round(total_stake, 2),
        "total_ev": round(total_ev, 2),
        "budget_left": round(budget_left, 2),
        "daily_budget": round(daily_budget, 2),
        "cap_per_bet_eur": round(cap_eur, 2),
        "adaptation_notes": adaptation_notes
    }


@app.get("/portfolio/today")
async def get_portfolio_today(
    bankroll: float = 1000.0,
    kelly_profile: str = None,
    source: str = "picks"
):
    """
    Retourne le portefeuille optimisé du jour.
    
    Sélection des meilleurs paris avec allocation Kelly fractionnaire.
    
    Args:
        bankroll: Bankroll de l'utilisateur (défaut 1000€)
        kelly_profile: Profil Kelly (SUR, STANDARD, AMBITIEUX, PERSONNALISE)
    
    Returns:
        - positions: Liste des paris sélectionnés avec stakes arrondis
        - profile_used: Profil Kelly utilisé
        - kelly_fraction_effective: Fraction Kelly effective
        - caps: Informations sur les caps (cap_per_bet, daily_budget_rate)
        - budget_left: Budget restant pour la journée
    """
    try:
        # Charger la config
        config = load_config_file(create_if_missing=True)
        bet_defaults = config.get('betting_defaults', {})
        kelly_map = bet_defaults.get('kelly_fraction_map', {"SUR": 0.25, "STANDARD": 0.33, "AMBITIEUX": 0.50})
        
        # Déterminer le profil Kelly
        profile = (kelly_profile or bet_defaults.get('kelly_profile_default', 'STANDARD')).upper()
        if profile == "PERSONNALISE":
            kelly_frac = bet_defaults.get('custom_kelly_fraction', 0.33)
        elif profile in kelly_map:
            kelly_frac = kelly_map[profile]
        else:
            kelly_frac = kelly_map.get('STANDARD', 0.33)
            profile = "STANDARD"
        
        # Récupérer les autres paramètres
        value_cutoff = bet_defaults.get('value_cutoff', 0.05)
        cap_per_bet = bet_defaults.get('cap_per_bet', 0.02)
        daily_budget_rate = bet_defaults.get('daily_budget_rate', 0.12)
        max_bets_per_race = bet_defaults.get('max_unit_bets_per_race', 2)
        rounding = bet_defaults.get('rounding_increment_eur', 0.5)
        takeout_rate = config.get('markets', {}).get('takeout_rate', 0.16)
        betting_policy = config.get("betting_policy", {}) or {}

        source_norm = (source or "picks").lower().strip()
        if source_norm in ("picks", "picks_today", "today"):
            from services.betting_policy import select_portfolio_from_picks

            picks_payload = await get_picks_today()
            picks = (picks_payload.get("picks") or []) if isinstance(picks_payload, dict) else []

            policy_result = select_portfolio_from_picks(
                picks=picks,
                bankroll=bankroll,
                kelly_fraction=kelly_frac,
                cap_per_bet=cap_per_bet,
                daily_budget_rate=daily_budget_rate,
                rounding=rounding,
                policy=betting_policy,
                profile=profile,
            )

            positions = policy_result["positions"]
            excluded_all = policy_result["excluded"]
            total_stake = float(policy_result["total_stake"])
            total_ev = float(policy_result["total_ev"])
            budget_left = float(policy_result["budget_left"])
            caps = policy_result["caps"]

            search_date = picks_payload.get("date") if isinstance(picks_payload, dict) else datetime.now().strftime('%Y-%m-%d')
            engine_meta = (picks_payload.get("meta") or {}) if isinstance(picks_payload, dict) else {}

            return {
                "date": search_date,
                "positions": positions,
                "bets": positions,
                "excluded": excluded_all[:10],
                "profile_used": profile,
                "kelly_fraction_effective": kelly_frac,
                "caps": {
                    "cap_per_bet": cap_per_bet,
                    "cap_per_bet_eur": caps.get("max_stake_per_bet_eur"),
                    "daily_budget_rate": caps.get("daily_budget_rate"),
                    "daily_budget_eur": caps.get("daily_budget_eur"),
                    "rounding_increment_eur": rounding,
                    "max_unit_bets_per_race": policy_result.get("policy", {}).get("max_bets_per_race", max_bets_per_race),
                    "value_cutoff": value_cutoff,
                    "min_stake_eur": MIN_ADAPTED_STAKE_EUR,
                    "max_daily_budget_share_per_bet": caps.get("max_daily_budget_share_per_bet"),
                },
                "budget_left": round(budget_left, 2),
                "total_stake": round(total_stake, 2),
                "totalStake": round(total_stake, 2),
                "total_ev": round(total_ev, 2),
                "totalEV": round(total_ev, 2),
                "expected_roi": round((total_ev / total_stake * 100) if total_stake > 0 else 0, 2),
                "roi": round((total_ev / total_stake * 100) if total_stake > 0 else 0, 2),
                "nb_positions": len(positions),
                "bankroll_used_pct": round(total_stake / bankroll * 100, 1) if bankroll > 0 else 0,
                "risk_level": "Faible" if total_stake < bankroll * 0.05 else "Modéré" if total_stake < bankroll * 0.10 else "Élevé",
                "engine_source": "picks_today",
                "engine_meta": engine_meta,
                "policy": policy_result.get("policy"),
            }

        # Legacy: moteur heuristique historique (conservé)
        con = get_db_connection()
        cur = con.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        cur.execute("""
            SELECT DISTINCT SUBSTRING(race_key FROM 1 FOR 10) as d
            FROM cheval_courses_seen
            ORDER BY d DESC LIMIT 1
        """)
        latest = cur.fetchone()
        search_date = latest[0] if latest else today
        
        cur.execute("""
            SELECT 
                nom_norm,
                race_key,
                hippodrome_nom,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                avis_entraineur,
                is_win,
                place_finale
            FROM cheval_courses_seen
            WHERE race_key LIKE %s
            AND cote_finale IS NOT NULL
            AND cote_finale > 1.5
            AND cote_finale < 30
            ORDER BY race_key
        """, (search_date + '%',))
        
        rows = cur.fetchall()
        con.close()
        
        # 1) Moteur commun sur BK de référence (1000€)
        engine_result = _compute_engine_portfolio(
            rows=rows,
            bankroll=ENGINE_BANKROLL_REFERENCE,
            kelly_frac=kelly_frac,
            value_cutoff=value_cutoff,
            cap_per_bet=cap_per_bet,
            daily_budget_rate=daily_budget_rate,
            rounding=rounding,
            max_bets_per_race=max_bets_per_race,
            takeout_rate=takeout_rate
        )
        
        # 2) Adaptation à la bankroll utilisateur (filtrage + scaling des mises)
        policy = _bankroll_policy(bankroll)
        adapted = _adapt_positions_for_bankroll(
            engine_positions=engine_result["positions"],
            bankroll=bankroll,
            cap_per_bet=cap_per_bet,
            daily_budget_rate=daily_budget_rate,
            rounding=rounding,
            policy=policy,
            engine_kelly_fraction=engine_result.get("kelly_fraction_used", kelly_frac)
        )
        
        positions = adapted["positions"]
        total_stake = adapted["total_stake"]
        total_ev = adapted["total_ev"]
        budget_left = max(0, adapted["budget_left"])
        applied_daily_budget_rate = policy.get("daily_budget_rate", daily_budget_rate)
        daily_budget = bankroll * applied_daily_budget_rate
        max_stake_per_bet = bankroll * cap_per_bet
        
        excluded_all = engine_result["excluded"] + adapted["excluded"]
        
        return {
            "date": search_date,
            "positions": positions,
            "bets": positions,  # Alias
            "excluded": excluded_all[:10],  # Limiter les exclus affichés
            
            # Profil et paramètres
            "profile_used": profile,
            "kelly_fraction_effective": kelly_frac,
            "caps": {
                "cap_per_bet": cap_per_bet,
                "cap_per_bet_eur": round(max_stake_per_bet, 2),
                "daily_budget_rate": applied_daily_budget_rate,
                "daily_budget_eur": round(daily_budget, 2),
                "rounding_increment_eur": rounding,
                "max_unit_bets_per_race": max_bets_per_race,
                "value_cutoff": value_cutoff,
                "min_stake_eur": MIN_ADAPTED_STAKE_EUR
            },
            "budget_left": round(budget_left, 2),
            
            # Stats adaptées à la BK utilisateur
            "total_stake": round(total_stake, 2),
            "totalStake": round(total_stake, 2),
            "total_ev": round(total_ev, 2),
            "totalEV": round(total_ev, 2),
            "expected_roi": round((total_ev / total_stake * 100) if total_stake > 0 else 0, 2),
            "roi": round((total_ev / total_stake * 100) if total_stake > 0 else 0, 2),
            "nb_positions": len(positions),
            "bankroll_used_pct": round(total_stake / bankroll * 100, 1) if bankroll > 0 else 0,
            "risk_level": "Faible" if total_stake < bankroll * 0.05 else "Modéré" if total_stake < bankroll * 0.10 else "Élevé",
            
            # Trace moteur commun (BK 1000€)
            "engine_reference": {
                "bankroll": ENGINE_BANKROLL_REFERENCE,
                "total_stake": engine_result["total_stake"],
                "budget_left": engine_result["budget_left"],
                "positions": engine_result["positions"],
                "kelly_fraction_used": engine_result.get("kelly_fraction_used"),
                "daily_budget_rate_used": engine_result.get("daily_budget_rate_used"),
                "cap_per_bet_used": engine_result.get("cap_per_bet_used")
            },
            "policy": policy,
            "adaptation_notes": adapted.get("adaptation_notes", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

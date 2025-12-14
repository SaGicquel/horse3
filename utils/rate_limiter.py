# -*- coding: utf-8 -*-
"""
Rate Limiter Adaptatif pour les scrapers PMU

Ajuste automatiquement le délai entre les requêtes en fonction:
- Des erreurs rencontrées (augmente le délai)
- Des succès consécutifs (diminue le délai progressivement)
- Des codes HTTP 429 (Too Many Requests)

Usage:
    limiter = AdaptiveRateLimiter(min_delay=0.2, max_delay=10.0)
    
    for url in urls:
        limiter.wait()
        response = requests.get(url)
        
        if response.ok:
            limiter.success()
        else:
            limiter.error(response.status_code)
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterStats:
    """Statistiques du rate limiter"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    throttled_requests: int = 0  # HTTP 429
    total_wait_time: float = 0.0
    current_delay: float = 0.0


class AdaptiveRateLimiter:
    """
    Rate limiter adaptatif qui ajuste le délai entre les requêtes.
    
    Attributes:
        min_delay: Délai minimum entre les requêtes (secondes)
        max_delay: Délai maximum (en cas d'erreurs répétées)
        backoff_factor: Multiplicateur en cas d'erreur
        recovery_factor: Diviseur pour réduire le délai après succès
        consecutive_success_threshold: Nombre de succès avant réduction
    """
    
    def __init__(
        self,
        min_delay: float = 0.2,
        max_delay: float = 10.0,
        backoff_factor: float = 1.5,
        recovery_factor: float = 0.9,
        consecutive_success_threshold: int = 5
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.consecutive_success_threshold = consecutive_success_threshold
        
        self._current_delay = min_delay
        self._consecutive_success = 0
        self._lock = threading.Lock()
        self._last_request_time: Optional[float] = None
        
        # Statistiques
        self._stats = RateLimiterStats(current_delay=min_delay)
    
    @property
    def current_delay(self) -> float:
        """Retourne le délai actuel"""
        with self._lock:
            return self._current_delay
    
    @property
    def stats(self) -> RateLimiterStats:
        """Retourne les statistiques actuelles"""
        with self._lock:
            self._stats.current_delay = self._current_delay
            return self._stats
    
    def wait(self) -> float:
        """
        Attend le délai nécessaire avant la prochaine requête.
        
        Returns:
            Temps effectivement attendu (secondes)
        """
        with self._lock:
            now = time.time()
            
            if self._last_request_time is not None:
                elapsed = now - self._last_request_time
                wait_time = max(0, self._current_delay - elapsed)
            else:
                wait_time = 0
            
            if wait_time > 0:
                time.sleep(wait_time)
                self._stats.total_wait_time += wait_time
            
            self._last_request_time = time.time()
            self._stats.total_requests += 1
            
            return wait_time
    
    def success(self):
        """
        Signale une requête réussie.
        Réduit progressivement le délai après plusieurs succès consécutifs.
        """
        with self._lock:
            self._consecutive_success += 1
            self._stats.successful_requests += 1
            
            # Réduire le délai après suffisamment de succès
            if self._consecutive_success >= self.consecutive_success_threshold:
                old_delay = self._current_delay
                self._current_delay = max(
                    self.min_delay,
                    self._current_delay * self.recovery_factor
                )
                self._consecutive_success = 0
                
                if self._current_delay < old_delay:
                    logger.debug(
                        f"Rate limiter: délai réduit {old_delay:.2f}s → {self._current_delay:.2f}s"
                    )
    
    def error(self, status_code: Optional[int] = None):
        """
        Signale une erreur.
        Augmente le délai, surtout en cas de HTTP 429.
        
        Args:
            status_code: Code HTTP de la réponse (optionnel)
        """
        with self._lock:
            self._consecutive_success = 0
            self._stats.failed_requests += 1
            
            old_delay = self._current_delay
            
            if status_code == 429:
                # Too Many Requests - augmentation forte
                self._stats.throttled_requests += 1
                self._current_delay = min(
                    self.max_delay,
                    self._current_delay * 3
                )
                logger.warning(
                    f"Rate limiter: HTTP 429, délai augmenté {old_delay:.2f}s → {self._current_delay:.2f}s"
                )
            elif status_code and status_code >= 500:
                # Erreur serveur - augmentation modérée
                self._current_delay = min(
                    self.max_delay,
                    self._current_delay * 2
                )
                logger.warning(
                    f"Rate limiter: Erreur serveur {status_code}, délai {old_delay:.2f}s → {self._current_delay:.2f}s"
                )
            else:
                # Autre erreur - augmentation standard
                self._current_delay = min(
                    self.max_delay,
                    self._current_delay * self.backoff_factor
                )
                logger.debug(
                    f"Rate limiter: Erreur, délai {old_delay:.2f}s → {self._current_delay:.2f}s"
                )
    
    def reset(self):
        """Remet le rate limiter à son état initial"""
        with self._lock:
            self._current_delay = self.min_delay
            self._consecutive_success = 0
            self._last_request_time = None
            logger.info("Rate limiter: réinitialisé")
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveRateLimiter("
            f"delay={self._current_delay:.2f}s, "
            f"min={self.min_delay}s, max={self.max_delay}s, "
            f"success_streak={self._consecutive_success})"
        )


# Singleton global pour usage simple
_default_limiter: Optional[AdaptiveRateLimiter] = None


def get_default_limiter() -> AdaptiveRateLimiter:
    """Retourne le rate limiter par défaut (singleton)"""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = AdaptiveRateLimiter()
    return _default_limiter


def rate_limited_request(func):
    """
    Décorateur pour ajouter le rate limiting à une fonction de requête.
    
    Usage:
        @rate_limited_request
        def fetch_data(url):
            return requests.get(url)
    """
    def wrapper(*args, **kwargs):
        limiter = get_default_limiter()
        limiter.wait()
        
        try:
            result = func(*args, **kwargs)
            
            # Vérifier si c'est une Response requests
            if hasattr(result, 'status_code'):
                if result.ok:
                    limiter.success()
                else:
                    limiter.error(result.status_code)
            else:
                limiter.success()
            
            return result
            
        except Exception as e:
            limiter.error()
            raise
    
    return wrapper


if __name__ == "__main__":
    # Test du rate limiter
    logging.basicConfig(level=logging.DEBUG)
    
    limiter = AdaptiveRateLimiter(min_delay=0.1, max_delay=5.0)
    
    print("Test du rate limiter adaptatif")
    print("-" * 40)
    
    # Simuler des succès
    for i in range(10):
        wait = limiter.wait()
        print(f"Requête {i+1}: attendu {wait:.3f}s, délai actuel: {limiter.current_delay:.3f}s")
        limiter.success()
    
    print("\n--- Simulation d'erreurs ---\n")
    
    # Simuler des erreurs
    for i in range(3):
        limiter.wait()
        limiter.error(500)
        print(f"Erreur {i+1}: délai actuel: {limiter.current_delay:.3f}s")
    
    print("\n--- Simulation HTTP 429 ---\n")
    
    limiter.wait()
    limiter.error(429)
    print(f"Après 429: délai actuel: {limiter.current_delay:.3f}s")
    
    print("\n--- Statistiques ---")
    print(limiter.stats)

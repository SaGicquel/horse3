# -*- coding: utf-8 -*-
"""
Circuit Breaker pour les scrapers PMU

Protège contre les échecs en cascade lorsqu'un service est indisponible.
Trois états:
- CLOSED: Normal, les requêtes passent
- OPEN: Service indisponible, les requêtes sont bloquées
- HALF-OPEN: Tentative de récupération, une requête test passe

Usage:
    breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
    
    if breaker.can_execute():
        try:
            response = requests.get(url)
            breaker.record_success()
        except Exception:
            breaker.record_failure()
    else:
        print("Circuit ouvert, service indisponible")
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """États du circuit breaker"""
    CLOSED = "closed"       # Normal - requêtes passent
    OPEN = "open"           # Bloqué - requêtes échouent immédiatement  
    HALF_OPEN = "half-open" # Test - une requête peut passer


@dataclass
class CircuitBreakerStats:
    """Statistiques du circuit breaker"""
    state: str = "closed"
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Appels bloqués car circuit ouvert
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    times_opened: int = 0


class CircuitBreaker:
    """
    Circuit breaker pour protéger contre les services défaillants.
    
    Attributes:
        failure_threshold: Nombre d'échecs consécutifs avant ouverture
        success_threshold: Nombre de succès pour fermer le circuit (HALF-OPEN → CLOSED)
        reset_timeout: Temps en secondes avant de passer à HALF-OPEN
        exclude_exceptions: Types d'exceptions à ne pas compter comme échecs
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        reset_timeout: float = 60.0,
        exclude_exceptions: tuple = ()
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.reset_timeout = reset_timeout
        self.exclude_exceptions = exclude_exceptions
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        
        # Statistiques
        self._stats = CircuitBreakerStats()
    
    @property
    def state(self) -> CircuitState:
        """Retourne l'état actuel du circuit"""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Retourne les statistiques"""
        with self._lock:
            self._stats.state = self._state.value
            self._stats.consecutive_failures = self._failure_count
            self._stats.last_failure_time = self._last_failure_time
            return self._stats
    
    def _check_state_transition(self):
        """Vérifie et effectue les transitions d'état automatiques"""
        if self._state == CircuitState.OPEN:
            # Vérifier si on peut passer à HALF-OPEN
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker: OPEN → HALF-OPEN après {elapsed:.1f}s"
                    )
    
    def can_execute(self) -> bool:
        """
        Vérifie si une requête peut être exécutée.
        
        Returns:
            True si la requête peut passer, False sinon
        """
        with self._lock:
            self._check_state_transition()
            self._stats.total_calls += 1
            
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.HALF_OPEN:
                # En HALF-OPEN, on laisse passer une requête test
                return True
            
            # OPEN - bloquer la requête
            self._stats.rejected_calls += 1
            return False
    
    def record_success(self):
        """Enregistre un succès"""
        with self._lock:
            self._stats.successful_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    # Assez de succès, fermer le circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker: HALF-OPEN → CLOSED")
            
            elif self._state == CircuitState.CLOSED:
                # Réinitialiser le compteur d'échecs
                self._failure_count = 0
    
    def record_failure(self, exception: Optional[Exception] = None):
        """
        Enregistre un échec.
        
        Args:
            exception: L'exception qui a causé l'échec (optionnel)
        """
        # Vérifier si l'exception doit être exclue
        if exception and self.exclude_exceptions:
            if isinstance(exception, self.exclude_exceptions):
                logger.debug(f"Circuit breaker: exception {type(exception)} ignorée")
                return
        
        with self._lock:
            self._stats.failed_calls += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Échec pendant le test, rouvrir le circuit
                self._state = CircuitState.OPEN
                self._stats.times_opened += 1
                logger.warning("Circuit breaker: HALF-OPEN → OPEN (échec test)")
            
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    # Trop d'échecs, ouvrir le circuit
                    self._state = CircuitState.OPEN
                    self._stats.times_opened += 1
                    logger.warning(
                        f"Circuit breaker: CLOSED → OPEN après {self._failure_count} échecs"
                    )
    
    def reset(self):
        """Remet le circuit breaker à son état initial"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info("Circuit breaker: réinitialisé → CLOSED")
    
    def force_open(self):
        """Force l'ouverture du circuit (pour maintenance)"""
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            logger.warning("Circuit breaker: forcé → OPEN")
    
    def is_open(self) -> bool:
        """Raccourci pour vérifier si le circuit est ouvert"""
        return self.state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Raccourci pour vérifier si le circuit est fermé"""
        return self.state == CircuitState.CLOSED
    
    def __call__(self, func: Callable) -> Callable:
        """
        Décorateur pour protéger une fonction avec le circuit breaker.
        
        Usage:
            breaker = CircuitBreaker()
            
            @breaker
            def fetch_data():
                return requests.get(url)
        """
        def wrapper(*args, **kwargs) -> Any:
            if not self.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker ouvert, réessayez dans {self.reset_timeout}s"
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    def __repr__(self) -> str:
        return (
            f"CircuitBreaker("
            f"state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold}, "
            f"timeout={self.reset_timeout}s)"
        )


class CircuitBreakerOpenError(Exception):
    """Exception levée quand le circuit est ouvert"""
    pass


# Singleton global pour usage simple
_default_breaker: Optional[CircuitBreaker] = None


def get_default_breaker() -> CircuitBreaker:
    """Retourne le circuit breaker par défaut (singleton)"""
    global _default_breaker
    if _default_breaker is None:
        _default_breaker = CircuitBreaker()
    return _default_breaker


if __name__ == "__main__":
    # Test du circuit breaker
    logging.basicConfig(level=logging.INFO)
    
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5)
    
    print("Test du circuit breaker")
    print("-" * 40)
    
    # Simuler des succès
    print("\n--- Succès ---")
    for i in range(5):
        if breaker.can_execute():
            print(f"Requête {i+1}: OK")
            breaker.record_success()
        else:
            print(f"Requête {i+1}: BLOQUÉE")
    
    print(f"État: {breaker.state.value}")
    
    # Simuler des échecs
    print("\n--- Échecs consécutifs ---")
    for i in range(5):
        if breaker.can_execute():
            print(f"Requête {i+1}: ÉCHEC")
            breaker.record_failure()
        else:
            print(f"Requête {i+1}: BLOQUÉE (circuit ouvert)")
    
    print(f"État: {breaker.state.value}")
    
    # Attendre le reset
    print(f"\n--- Attente du reset ({breaker.reset_timeout}s) ---")
    time.sleep(breaker.reset_timeout + 1)
    
    print(f"État après timeout: {breaker.state.value}")
    
    # Tester en half-open
    if breaker.can_execute():
        print("Requête test en HALF-OPEN: SUCCÈS")
        breaker.record_success()
    
    if breaker.can_execute():
        print("2ème requête: SUCCÈS")
        breaker.record_success()
    
    print(f"État final: {breaker.state.value}")
    print("\n--- Statistiques ---")
    print(breaker.stats)

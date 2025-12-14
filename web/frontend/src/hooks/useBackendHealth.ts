import { useState, useEffect, useCallback } from 'react';
import { env } from '../config/env';

export interface BackendHealth {
  status: 'healthy' | 'unhealthy' | 'checking' | 'unknown';
  service?: string;
  latencyMs?: number;
  lastCheck?: Date;
  error?: string;
}

interface UseBackendHealthOptions {
  /** Intervalle entre les vérifications automatiques (ms). 0 = désactivé */
  pollInterval?: number;
  /** Timeout pour la requête health (ms) */
  timeout?: number;
  /** Activer la vérification automatique au montage */
  checkOnMount?: boolean;
}

const DEFAULT_OPTIONS: UseBackendHealthOptions = {
  pollInterval: 60000, // 1 minute
  timeout: 5000,
  checkOnMount: true,
};

/**
 * Hook pour surveiller la santé du backend
 * 
 * @example
 * ```tsx
 * const { health, isHealthy, checkHealth } = useBackendHealth();
 * 
 * if (!isHealthy) {
 *   return <Alert>Backend indisponible</Alert>;
 * }
 * ```
 */
export function useBackendHealth(options: UseBackendHealthOptions = {}) {
  const { pollInterval, timeout, checkOnMount } = { ...DEFAULT_OPTIONS, ...options };
  
  const [health, setHealth] = useState<BackendHealth>({
    status: 'unknown',
  });

  const checkHealth = useCallback(async (): Promise<BackendHealth> => {
    setHealth(prev => ({ ...prev, status: 'checking' }));
    
    const startTime = performance.now();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      // Utiliser /api/health ou /healthz selon la config nginx
      const healthUrl = `${env.API_BASE_URL.replace(/\/api$/, '')}/healthz`;
      
      const response = await fetch(healthUrl, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        },
      });
      
      clearTimeout(timeoutId);
      const latencyMs = Math.round(performance.now() - startTime);
      
      if (response.ok) {
        const data = await response.json().catch(() => ({}));
        const newHealth: BackendHealth = {
          status: 'healthy',
          service: data.service || 'horse-backend',
          latencyMs,
          lastCheck: new Date(),
        };
        setHealth(newHealth);
        return newHealth;
      } else {
        const newHealth: BackendHealth = {
          status: 'unhealthy',
          latencyMs,
          lastCheck: new Date(),
          error: `HTTP ${response.status}`,
        };
        setHealth(newHealth);
        return newHealth;
      }
    } catch (error) {
      clearTimeout(timeoutId);
      const latencyMs = Math.round(performance.now() - startTime);
      
      const isTimeout = error instanceof Error && error.name === 'AbortError';
      const newHealth: BackendHealth = {
        status: 'unhealthy',
        latencyMs,
        lastCheck: new Date(),
        error: isTimeout ? 'Timeout' : (error instanceof Error ? error.message : 'Network error'),
      };
      setHealth(newHealth);
      return newHealth;
    }
  }, [timeout]);

  // Vérification au montage
  useEffect(() => {
    if (checkOnMount) {
      checkHealth();
    }
  }, [checkOnMount, checkHealth]);

  // Polling automatique
  useEffect(() => {
    if (!pollInterval || pollInterval <= 0) return;
    
    const intervalId = setInterval(checkHealth, pollInterval);
    return () => clearInterval(intervalId);
  }, [pollInterval, checkHealth]);

  return {
    health,
    isHealthy: health.status === 'healthy',
    isChecking: health.status === 'checking',
    checkHealth,
  };
}

export default useBackendHealth;

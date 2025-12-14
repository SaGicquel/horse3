/**
 * Configuration d'environnement centralisée pour le frontend
 * 
 * Priorité de résolution de l'URL API :
 * 1. window.__APP_CONFIG__ (injection runtime via /app-config.js)
 * 2. import.meta.env.VITE_API_URL (variable d'env Vite au build)
 * 3. /api (production Docker avec nginx proxy)
 * 4. http://localhost:8000/api (développement local)
 */

// Types pour la configuration
interface AppConfig {
  apiBaseUrl?: string;
  environment?: 'development' | 'production' | 'staging';
  version?: string;
}

// Extension de Window pour la config injectée
declare global {
  interface Window {
    __APP_CONFIG__?: AppConfig;
  }
}

/**
 * Détermine l'URL de base de l'API en fonction de l'environnement
 */
const resolveApiBaseUrl = (): string => {
  // 1. Config injectée au runtime (Docker avec window.__APP_CONFIG__)
  if (typeof window !== 'undefined' && window.__APP_CONFIG__?.apiBaseUrl) {
    return window.__APP_CONFIG__.apiBaseUrl.replace(/\/$/, '');
  }

  // 2. Variable d'environnement Vite (définie au build)
  if (import.meta.env.VITE_API_URL) {
    return (import.meta.env.VITE_API_URL as string).replace(/\/$/, '');
  }

  // 3. Production Docker : utiliser le proxy nginx /api
  if (typeof window !== 'undefined') {
    const { port, hostname } = window.location;
    const isProduction = !port || port === '80' || port === '443';
    const isNotLocalDev = hostname !== 'localhost' || isProduction;
    
    if (isNotLocalDev) {
      return '/api';
    }
  }

  // 4. Fallback développement local
  return 'http://localhost:8000/api';
};

/**
 * Configuration exportée
 */
export const env = {
  /** URL de base de l'API (sans trailing slash) */
  API_BASE_URL: resolveApiBaseUrl(),
  
  /** URL racine du backend (sans /api) */
  get API_BASE(): string {
    return this.API_BASE_URL.replace(/\/api$/, '') || 'http://localhost:8000';
  },
  
  /** Mode développement */
  isDevelopment: import.meta.env.DEV ?? false,
  
  /** Mode production */
  isProduction: import.meta.env.PROD ?? true,
  
  /** Version de l'app */
  version: import.meta.env.VITE_APP_VERSION || '1.0.0',
} as const;

/**
 * Log de la configuration au démarrage (en dev uniquement)
 */
if (env.isDevelopment) {
  const source = (typeof window !== 'undefined' && window.__APP_CONFIG__?.apiBaseUrl)
    ? 'runtime-injection'
    : import.meta.env.VITE_API_URL 
      ? 'vite-env' 
      : 'auto-detect';

  console.log('[env] Configuration chargée:', {
    API_BASE_URL: env.API_BASE_URL,
    isDevelopment: env.isDevelopment,
    source,
  });
}

export default env;

/**
 * Système de cache API pour améliorer les performances
 * Cache en mémoire avec expiration et déduplication des requêtes
 */

// Cache en mémoire avec TTL
const cache = new Map();
const pendingRequests = new Map();

// Durées de cache par type de données (en ms)
const CACHE_TTL = {
  dashboard: 60 * 1000,        // 1 minute - données du dashboard
  monitoring: 30 * 1000,       // 30 secondes - monitoring temps réel
  analytics: 5 * 60 * 1000,    // 5 minutes - analytics (données lourdes)
  chevaux: 2 * 60 * 1000,      // 2 minutes - liste des chevaux
  chevalDetails: 5 * 60 * 1000,// 5 minutes - détails d'un cheval
  courses: 2 * 60 * 1000,      // 2 minutes - courses
  default: 60 * 1000,          // 1 minute par défaut
};

/**
 * Génère une clé de cache unique pour une requête
 */
export const getCacheKey = (endpoint, params = {}) => {
  const sortedParams = Object.keys(params)
    .sort()
    .map(k => `${k}=${params[k]}`)
    .join('&');
  return `${endpoint}${sortedParams ? '?' + sortedParams : ''}`;
};

/**
 * Vérifie si une entrée de cache est encore valide
 */
const isValid = (entry) => {
  if (!entry) return false;
  return Date.now() - entry.timestamp < entry.ttl;
};

/**
 * Récupère une valeur du cache
 */
export const getFromCache = (key) => {
  const entry = cache.get(key);
  if (isValid(entry)) {
    return entry.data;
  }
  // Nettoyer l'entrée expirée
  cache.delete(key);
  return null;
};

/**
 * Stocke une valeur dans le cache
 */
export const setInCache = (key, data, type = 'default') => {
  const ttl = CACHE_TTL[type] || CACHE_TTL.default;
  cache.set(key, {
    data,
    timestamp: Date.now(),
    ttl,
  });
};

/**
 * Invalide une entrée ou tout le cache
 */
export const invalidateCache = (key = null) => {
  if (key) {
    cache.delete(key);
    // Invalider aussi les entrées qui commencent par cette clé
    for (const k of cache.keys()) {
      if (k.startsWith(key)) {
        cache.delete(k);
      }
    }
  } else {
    cache.clear();
  }
};

/**
 * Wrapper pour les appels API avec cache et déduplication
 * Empêche les requêtes multiples simultanées pour la même ressource
 */
export const cachedFetch = async (fetchFn, cacheKey, cacheType = 'default', forceRefresh = false) => {
  // Vérifier le cache sauf si refresh forcé
  if (!forceRefresh) {
    const cachedData = getFromCache(cacheKey);
    if (cachedData !== null) {
      return cachedData;
    }
  }

  // Vérifier si une requête est déjà en cours pour cette ressource
  if (pendingRequests.has(cacheKey)) {
    return pendingRequests.get(cacheKey);
  }

  // Créer la promesse de requête
  const requestPromise = fetchFn()
    .then((data) => {
      setInCache(cacheKey, data, cacheType);
      pendingRequests.delete(cacheKey);
      return data;
    })
    .catch((error) => {
      pendingRequests.delete(cacheKey);
      throw error;
    });

  // Stocker la promesse pour déduplication
  pendingRequests.set(cacheKey, requestPromise);
  return requestPromise;
};

/**
 * Précharge les données critiques en arrière-plan
 */
export const prefetchCriticalData = async (fetchFunctions) => {
  // Utiliser requestIdleCallback si disponible, sinon setTimeout
  const scheduleTask = window.requestIdleCallback || ((cb) => setTimeout(cb, 100));
  
  scheduleTask(() => {
    Promise.allSettled(fetchFunctions.map(fn => fn()));
  });
};

/**
 * Nettoie les entrées de cache expirées
 */
export const cleanExpiredCache = () => {
  for (const [key, entry] of cache.entries()) {
    if (!isValid(entry)) {
      cache.delete(key);
    }
  }
};

// Nettoyage périodique du cache (toutes les 5 minutes)
if (typeof window !== 'undefined') {
  setInterval(cleanExpiredCache, 5 * 60 * 1000);
}

export default {
  getFromCache,
  setInCache,
  invalidateCache,
  cachedFetch,
  prefetchCriticalData,
  getCacheKey,
};

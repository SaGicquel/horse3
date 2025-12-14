import axios from 'axios';
import { cachedFetch, getCacheKey, invalidateCache } from './apiCache';
import { env } from '../config/env';

// Utiliser la configuration centralisée depuis env.ts
const API_BASE_URL = env.API_BASE_URL;

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }
    if (error?.code === 'ERR_NETWORK' || error?.message?.includes('Network Error')) {
      return Promise.reject(new Error('API indisponible. Vérifiez l\'URL et que le backend est démarré.'));
    }
    const message = error?.response?.data?.message || error.message || 'Une erreur est survenue';
    return Promise.reject(new Error(message));
  }
);

const withAuth = (token) => ({
  headers: token ? { Authorization: `Bearer ${token}` } : {},
});

export const authAPI = {
  register: async (payload) => {
    const response = await api.post('/auth/register', payload);
    return response.data;
  },
  login: async (payload) => {
    const response = await api.post('/auth/login', payload);
    return response.data;
  },
  me: async (token) => {
    const response = await api.get('/auth/me', withAuth(token));
    return response.data;
  }
};

export const betsAPI = {
  list: async (token) => {
    const response = await api.get('/bets', withAuth(token));
    return response.data;
  },
  create: async (payload, token) => {
    const response = await api.post('/bets', payload, withAuth(token));
    return response.data;
  },
  update: async (betId, payload, token) => {
    const response = await api.patch(`/bets/${betId}`, payload, withAuth(token));
    return response.data;
  },
  delete: async (betId, token) => {
    const response = await api.delete(`/bets/${betId}`, withAuth(token));
    return response.data;
  },
  refreshResult: async (betId, token) => {
    const response = await api.post(`/bets/${betId}/refresh`, {}, withAuth(token));
    return response.data;
  },
  summary: async (token) => {
    const response = await api.get('/bets/summary', withAuth(token));
    return response.data;
  }
};

export const dashboardAPI = {
  getDashboardData: async (token, signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/dashboard', { auth: token ? token.slice(0, 10) : 'public' });
    return cachedFetch(
      async () => {
        const response = await api.get('/dashboard', {
          signal,
          ...(token ? withAuth(token) : {})
        });
        return response.data;
      },
      cacheKey,
      'dashboard',
      forceRefresh
    );
  },
  
  getMonitoringData: async (token, signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/monitoring', { auth: token ? token.slice(0, 10) : 'public' });
    return cachedFetch(
      async () => {
        const response = await api.get('/monitoring', {
          signal,
          ...(token ? withAuth(token) : {})
        });
        return response.data;
      },
      cacheKey,
      'monitoring',
      forceRefresh
    );
  },
  
  refresh: () => {
    invalidateCache('/dashboard');
    invalidateCache('/monitoring');
  },
};

export const chevauxAPI = {
  getAllChevaux: async (limit = 50, offset = 0, sortBy = null, sortOrder = 'asc', search = null, signal, forceRefresh = false) => {
    let url = `/chevaux?limit=${limit}&offset=${offset}`;
    if (sortBy) {
      url += `&sort_by=${sortBy}&sort_order=${sortOrder}`;
    }
    if (search && search.trim() !== '') {
      url += `&search=${encodeURIComponent(search.trim())}`;
    }
    const cacheKey = getCacheKey(url);
    return cachedFetch(
      async () => {
        const response = await api.get(url, { signal });
        return response.data;
      },
      cacheKey,
      'chevaux',
      forceRefresh
    );
  },
  
  getChevalDetails: async (nom, signal, forceRefresh = false) => {
    const cacheKey = getCacheKey(`/chevaux/${nom}`);
    return cachedFetch(
      async () => {
        const response = await api.get(`/chevaux/${nom}`, { signal });
        return response.data;
      },
      cacheKey,
      'chevalDetails',
      forceRefresh
    );
  },
};

export const coursesAPI = {
  getCoursesVues: async (limit = 50, offset = 0, sortBy = null, sortOrder = 'asc', search = null, signal, forceRefresh = false) => {
    let url = `/courses-vues?limit=${limit}&offset=${offset}`;
    if (sortBy) {
      url += `&sort_by=${sortBy}&sort_order=${sortOrder}`;
    }
    if (search && search.trim() !== '') {
      url += `&search=${encodeURIComponent(search.trim())}`;
    }
    const cacheKey = getCacheKey(url);
    return cachedFetch(
      async () => {
        const response = await api.get(url, { signal });
        return response.data;
      },
      cacheKey,
      'courses',
      forceRefresh
    );
  },
};

export const analyticsAPI = {
  getAnalyticsChevaux: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/chevaux');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/chevaux', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },
  
  getAnalyticsJockeys: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/jockeys');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/jockeys', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },
  
  getAnalyticsEntraineurs: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/entraineurs');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/entraineurs', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },
  
  getAnalyticsHippodromes: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/hippodromes');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/hippodromes', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },
  
  getAnalyticsEvolution: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/evolution');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/evolution', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },
  
  getTauxParRace: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/taux-par-race');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/taux-par-race', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },

  getAnalyticsOdds: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/odds');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/odds', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },

  getAnalyticsDistance: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/distance');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/distance', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },

  getAnalyticsAge: async (signal, forceRefresh = false) => {
    const cacheKey = getCacheKey('/analytics/age');
    return cachedFetch(
      async () => {
        const response = await api.get('/analytics/age', { signal });
        return response.data;
      },
      cacheKey,
      'analytics',
      forceRefresh
    );
  },
  
  refresh: () => {
    invalidateCache('/analytics');
  },
};

export default api;

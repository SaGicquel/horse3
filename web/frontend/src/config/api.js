// Configuration centralisée de l'API
// Réexport depuis env.ts pour rétrocompatibilité
// Utilise window.__APP_CONFIG__ > VITE_API_URL > /api > localhost:8000

import { env } from './env';

export const API_BASE_URL = env.API_BASE_URL;
export const API_BASE = env.API_BASE;

export default API_BASE_URL;

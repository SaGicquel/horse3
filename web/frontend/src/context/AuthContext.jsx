/**
 * AuthContext - Contexte d'authentification global
 * Gère l'état de connexion utilisateur dans toute l'application
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authAPI } from '../services/api';

// Clé de stockage unifiée
const TOKEN_KEY = 'hrp_token';
const USER_KEY = 'hrp_user';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    // Vérifier si l'utilisateur est authentifié
    const isAuthenticated = Boolean(token && user);

    // Charger la session depuis localStorage au démarrage
    useEffect(() => {
        const storedToken = localStorage.getItem(TOKEN_KEY);
        const storedUser = localStorage.getItem(USER_KEY);

        if (storedToken && storedUser) {
            try {
                setToken(storedToken);
                setUser(JSON.parse(storedUser));
            } catch (e) {
                // Données corrompues, nettoyer
                localStorage.removeItem(TOKEN_KEY);
                localStorage.removeItem(USER_KEY);
            }
        }

        // Sync avec l'ancienne clé authToken si elle existe
        const legacyToken = localStorage.getItem('authToken');
        if (legacyToken && !storedToken) {
            localStorage.setItem(TOKEN_KEY, legacyToken);
            setToken(legacyToken);
        }

        setIsLoading(false);
    }, []);

    // Synchroniser avec localStorage quand le token change
    useEffect(() => {
        if (token) {
            localStorage.setItem(TOKEN_KEY, token);
            // Sync avec l'ancienne clé pour compatibilité
            localStorage.setItem('authToken', token);
        }
    }, [token]);

    useEffect(() => {
        if (user) {
            localStorage.setItem(USER_KEY, JSON.stringify(user));
        }
    }, [user]);

    // Fonction de connexion
    const login = useCallback(async (email, password) => {
        setError(null);
        setIsLoading(true);
        try {
            const data = await authAPI.login({ email, password });
            setToken(data.token);
            setUser(data.user);
            return { success: true, user: data.user };
        } catch (err) {
            const message = err.message || 'Erreur de connexion';
            setError(message);
            return { success: false, error: message };
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Fonction d'inscription
    const register = useCallback(async (email, password, displayName) => {
        setError(null);
        setIsLoading(true);
        try {
            const payload = { email, password };
            if (displayName?.trim()) {
                payload.display_name = displayName.trim();
            }
            const data = await authAPI.register(payload);
            setToken(data.token);
            setUser(data.user);
            return { success: true, user: data.user };
        } catch (err) {
            const message = err.message || 'Erreur lors de l\'inscription';
            setError(message);
            return { success: false, error: message };
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Fonction de déconnexion
    const logout = useCallback(() => {
        setToken(null);
        setUser(null);
        setError(null);
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        localStorage.removeItem('authToken'); // Nettoyer aussi l'ancienne clé
    }, []);

    // Vérifier la validité du token (appel API)
    const checkAuth = useCallback(async () => {
        const storedToken = localStorage.getItem(TOKEN_KEY);
        if (!storedToken) {
            setIsLoading(false);
            return false;
        }

        try {
            const userData = await authAPI.me(storedToken);
            setUser(userData);
            setToken(storedToken);
            return true;
        } catch (err) {
            // Token invalide, nettoyer
            logout();
            return false;
        } finally {
            setIsLoading(false);
        }
    }, [logout]);

    // Effacer l'erreur
    const clearError = useCallback(() => {
        setError(null);
    }, []);

    const value = {
        // État
        user,
        token,
        isAuthenticated,
        isLoading,
        error,

        // Actions
        login,
        register,
        logout,
        checkAuth,
        clearError,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

// Hook pour utiliser le contexte d'auth
export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth doit être utilisé dans un AuthProvider');
    }
    return context;
}

export default AuthContext;

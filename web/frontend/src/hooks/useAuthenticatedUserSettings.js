import { useState, useEffect } from 'react';

// Hook pour gérer les settings utilisateur authentifiés
export default function useAuthenticatedUserSettings() {
  const [userSettings, setUserSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // Fonction pour récupérer le token d'authentification
  const getAuthToken = () => {
    return localStorage.getItem('authToken');
  };

  // Fonction pour créer les headers avec authentification
  const getAuthHeaders = () => {
    const token = getAuthToken();
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` }),
    };
  };

  // Fonction pour charger les settings depuis l'API
  const loadUserSettings = async () => {
    const token = getAuthToken();
    if (!token) {
      setIsAuthenticated(false);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/user/settings`, {
        headers: getAuthHeaders(),
      });

      if (response.status === 401) {
        setIsAuthenticated(false);
        setError('Non authentifié');
        return;
      }

      if (!response.ok) {
        throw new Error('Impossible de charger les settings');
      }

      const settings = await response.json();
      setUserSettings(settings);
      setIsAuthenticated(true);
      setError(null);
    } catch (err) {
      console.error('Erreur chargement user settings:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fonction pour sauvegarder les settings
  const saveUserSettings = async (newSettings) => {
    const token = getAuthToken();
    if (!token) {
      throw new Error('Non authentifié');
    }

    try {
      const response = await fetch(`${API_BASE}/api/user/settings`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(newSettings),
      });

      if (response.status === 401) {
        setIsAuthenticated(false);
        throw new Error('Session expirée');
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Erreur sauvegarde');
      }

      setUserSettings(newSettings);
      
      // Émettre un événement pour synchroniser avec les autres composants
      window.dispatchEvent(new CustomEvent('userSettingsChanged', { 
        detail: newSettings 
      }));

      return await response.json();
    } catch (err) {
      console.error('Erreur sauvegarde user settings:', err);
      throw err;
    }
  };

  // Fonction pour mettre à jour un champ spécifique
  const updateSetting = async (field, value) => {
    if (!userSettings) return;
    
    const updatedSettings = { ...userSettings, [field]: value };
    await saveUserSettings(updatedSettings);
  };

  // Fonction pour mettre à jour plusieurs champs
  const updateSettings = async (updates) => {
    if (!userSettings) return;
    
    const updatedSettings = { ...userSettings, ...updates };
    await saveUserSettings(updatedSettings);
  };

  // Charger les settings au montage du composant
  useEffect(() => {
    loadUserSettings();
  }, []);

  // Écouter les changements de token d'authentification
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === 'authToken') {
        if (e.newValue) {
          loadUserSettings();
        } else {
          setIsAuthenticated(false);
          setUserSettings(null);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Getters pour les valeurs individuelles
  const bankroll = userSettings?.bankroll || 1000;
  const profilRisque = userSettings?.profil_risque || 'STANDARD';
  const kellyProfile = userSettings?.kelly_profile || 'STANDARD';
  const customKellyFraction = userSettings?.custom_kelly_fraction || 0.33;
  const valueCutoff = userSettings?.value_cutoff || 0.05;
  const capPerBet = userSettings?.cap_per_bet || 0.02;
  const dailyBudgetRate = userSettings?.daily_budget_rate || 0.12;
  const maxUnitBetsPerRace = userSettings?.max_unit_bets_per_race || 2;
  const roundingIncrementEur = userSettings?.rounding_increment_eur || 1.0;
  const perTicketRate = userSettings?.per_ticket_rate || 0.005;
  const maxPackRate = userSettings?.max_pack_rate || 0.03;
  const marketMode = userSettings?.market_mode || 'parimutuel';
  const takeoutRate = userSettings?.takeout_rate || 0.17;

  return {
    // État
    userSettings,
    loading,
    error,
    isAuthenticated,

    // Actions
    loadUserSettings,
    saveUserSettings,
    updateSetting,
    updateSettings,

    // Getters pour les valeurs individuelles
    bankroll,
    profilRisque,
    kellyProfile,
    customKellyFraction,
    valueCutoff,
    capPerBet,
    dailyBudgetRate,
    maxUnitBetsPerRace,
    roundingIncrementEur,
    perTicketRate,
    maxPackRate,
    marketMode,
    takeoutRate,

    // Setters pour les valeurs individuelles (pratique)
    setBankroll: (value) => updateSetting('bankroll', value),
    setProfilRisque: (value) => updateSetting('profil_risque', value),
    setKellyProfile: (value) => updateSetting('kelly_profile', value),
    setCustomKellyFraction: (value) => updateSetting('custom_kelly_fraction', value),
    setValueCutoff: (value) => updateSetting('value_cutoff', value),
    setCapPerBet: (value) => updateSetting('cap_per_bet', value),
    setDailyBudgetRate: (value) => updateSetting('daily_budget_rate', value),
    setMaxUnitBetsPerRace: (value) => updateSetting('max_unit_bets_per_race', value),
    setRoundingIncrementEur: (value) => updateSetting('rounding_increment_eur', value),
    setPerTicketRate: (value) => updateSetting('per_ticket_rate', value),
    setMaxPackRate: (value) => updateSetting('max_pack_rate', value),
    setMarketMode: (value) => updateSetting('market_mode', value),
    setTakeoutRate: (value) => updateSetting('takeout_rate', value),
  };
}
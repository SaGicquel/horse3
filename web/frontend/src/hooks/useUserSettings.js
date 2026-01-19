import { useState, useEffect } from 'react';

// Hook personnalisé pour gérer les paramètres utilisateur avec persistance
export const useUserSettings = () => {
  const [bankroll, setBankrollState] = useState(() => {
    const saved = localStorage.getItem('user_bankroll');
    return saved ? parseInt(saved) : 500;
  });

  const [profil, setProfilState] = useState(() => {
    const saved = localStorage.getItem('user_profil');
    return saved || 'STANDARD';
  });

  // Fonctions avec sauvegarde automatique
  const setBankroll = (newBankroll) => {
    setBankrollState(newBankroll);
    localStorage.setItem('user_bankroll', newBankroll.toString());

    // Déclencher un événement pour notifier les autres composants
    window.dispatchEvent(new CustomEvent('userSettingsChanged', {
      detail: { bankroll: newBankroll, profil }
    }));
  };

  const setProfil = (newProfil) => {
    setProfilState(newProfil);
    localStorage.setItem('user_profil', newProfil);

    // Déclencher un événement pour notifier les autres composants
    window.dispatchEvent(new CustomEvent('userSettingsChanged', {
      detail: { bankroll, profil: newProfil }
    }));
  };

  // Écouter les changements depuis d'autres pages
  useEffect(() => {
    const handleStorageChange = () => {
      const savedBankroll = localStorage.getItem('user_bankroll');
      const savedProfil = localStorage.getItem('user_profil');

      if (savedBankroll && parseInt(savedBankroll) !== bankroll) {
        setBankrollState(parseInt(savedBankroll));
      }
      if (savedProfil && savedProfil !== profil) {
        setProfilState(savedProfil);
      }
    };

    const handleCustomEvent = (event) => {
      const { bankroll: newBankroll, profil: newProfil } = event.detail;
      if (newBankroll !== bankroll) setBankrollState(newBankroll);
      if (newProfil !== profil) setProfilState(newProfil);
    };

    // Écouter les changements de localStorage (entre onglets)
    window.addEventListener('storage', handleStorageChange);

    // Écouter les changements depuis la même page
    window.addEventListener('userSettingsChanged', handleCustomEvent);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('userSettingsChanged', handleCustomEvent);
    };
  }, [bankroll, profil]);

  return {
    bankroll,
    profil,
    setBankroll,
    setProfil,
    // Fonction pour réinitialiser aux valeurs par défaut
    resetToDefaults: () => {
      setBankroll(500);
      setProfil('STANDARD');
    }
  };
};

export default useUserSettings;

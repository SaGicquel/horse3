import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Save } from 'lucide-react';

const UserSettingsNotification = () => {
  const [showNotification, setShowNotification] = useState(false);
  const [notificationData, setNotificationData] = useState(null);

  useEffect(() => {
    const handleUserSettingsChanged = (event) => {
      const { bankroll, profil } = event.detail;
      setNotificationData({ bankroll, profil });
      setShowNotification(true);

      // Masquer après 3 secondes
      setTimeout(() => {
        setShowNotification(false);
      }, 3000);
    };

    window.addEventListener('userSettingsChanged', handleUserSettingsChanged);

    return () => {
      window.removeEventListener('userSettingsChanged', handleUserSettingsChanged);
    };
  }, []);

  return (
    <AnimatePresence>
      {showNotification && notificationData && (
        <motion.div
          initial={{ opacity: 0, y: -100, scale: 0.8 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -100, scale: 0.8 }}
          transition={{ type: "spring", duration: 0.5 }}
          className="fixed top-4 right-4 z-50 bg-green-500/90 backdrop-blur-sm text-white px-4 py-3 rounded-lg shadow-lg flex items-center space-x-2"
        >
          <Check className="w-5 h-5" />
          <div>
            <div className="font-medium text-sm">Paramètres sauvegardés</div>
            <div className="text-xs opacity-90">
              {notificationData.bankroll}€ • {notificationData.profil}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default UserSettingsNotification;
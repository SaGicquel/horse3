/**
 * Composant ThemeToggle pour basculer entre mode clair et mode sombre
 * Avec effet Glassmorphism parfait
 * 
 * Utilise les variables CSS définies dans index.css :
 * - Mode clair : fond #F9FAFB, texte #111827
 * - Mode sombre : fond #0B0F1A, texte #E5E7EB
 * 
 * Contraste validé AA pour l'accessibilité
 */

import { Moon, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const ThemeToggle = () => {
  const [isDark, setIsDark] = useState(() => {
    // Récupérer la préférence depuis localStorage ou utiliser la préférence système
    const saved = localStorage.getItem('theme');
    if (saved) {
      return saved === 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    // Appliquer le thème au document
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  const toggleTheme = () => {
    setIsDark(!isDark);
  };

  return (
    <motion.button
      onClick={toggleTheme}
      className="
        flex items-center justify-center
        h-11 w-11 rounded-xl
        focus:outline-none
        transition-all duration-300
      "
      style={{
        background: 'rgba(var(--color-card-rgb), 0.5)',
        backdropFilter: 'blur(12px) saturate(150%)',
        WebkitBackdropFilter: 'blur(12px) saturate(150%)',
        border: '1px solid rgba(255, 255, 255, 0.15)',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
      }}
      whileHover={{ 
        scale: 1.05,
        boxShadow: '0 8px 24px rgba(var(--color-primary-rgb), 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.4)'
      }}
      whileTap={{ scale: 0.95 }}
      aria-label={isDark ? 'Activer le mode clair' : 'Activer le mode sombre'}
      title={isDark ? 'Mode clair' : 'Mode sombre'}
    >
      <motion.div
        initial={false}
        animate={{ rotate: isDark ? 180 : 0 }}
        transition={{ duration: 0.3 }}
      >
        {isDark ? (
          <Sun size={20} style={{ color: 'var(--color-primary)' }} />
        ) : (
          <Moon size={20} style={{ color: 'var(--color-primary)' }} />
        )}
      </motion.div>
    </motion.button>
  );
};

export default ThemeToggle;

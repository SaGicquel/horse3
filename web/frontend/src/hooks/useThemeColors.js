/**
 * Hook personnalisé pour accéder aux couleurs du thème
 *
 * Permet d'utiliser facilement les variables CSS du thème dans les composants React,
 * notamment pour les graphiques Recharts qui nécessitent des couleurs dynamiques.
 *
 * Usage :
 * ```jsx
 * import { useThemeColors } from '../hooks/useThemeColors';
 *
 * const MyComponent = () => {
 *   const colors = useThemeColors();
 *
 *   return <div style={{ color: colors.primary }}>Hello</div>;
 * };
 * ```
 */

import { useEffect, useState } from 'react';

export const useThemeColors = () => {
  const [colors, setColors] = useState({
    bg: '#F9FAFB',
    text: '#111827',
    card: '#FFFFFF',
    primary: '#9D3656',
    primaryHover: '#812C47',
    primaryLight: '#C86D8A',
    secondary: '#F5C3CE',
    border: 'rgba(157, 54, 86, 0.25)',
    borderHover: 'rgba(157, 54, 86, 0.35)',
    muted: '#6B7280',
    mutedLight: '#9CA3AF',
    success: '#2ED573',
    error: '#DC2626',
  });

  useEffect(() => {
    const updateColors = () => {
      const root = getComputedStyle(document.documentElement);
      setColors({
        bg: root.getPropertyValue('--color-bg').trim(),
        text: root.getPropertyValue('--color-text').trim(),
        card: root.getPropertyValue('--color-card').trim(),
        primary: root.getPropertyValue('--color-primary').trim(),
        primaryHover: root.getPropertyValue('--color-primary-hover').trim(),
        primaryLight: root.getPropertyValue('--color-primary-light').trim(),
        secondary: root.getPropertyValue('--color-secondary').trim(),
        border: root.getPropertyValue('--color-border').trim(),
        borderHover: root.getPropertyValue('--color-border-hover').trim(),
        muted: root.getPropertyValue('--color-muted').trim(),
        mutedLight: root.getPropertyValue('--color-muted-light').trim(),
        success: root.getPropertyValue('--color-success').trim(),
        error: root.getPropertyValue('--color-error').trim(),
      });
    };

    // Mise à jour initiale
    updateColors();

    // Observer les changements de classe 'dark' sur <html>
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === 'class') {
          updateColors();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  return colors;
};

/**
 * Hook pour savoir si le mode sombre est actif
 *
 * Usage :
 * ```jsx
 * const isDark = useIsDarkMode();
 * ```
 */
export const useIsDarkMode = () => {
  const [isDark, setIsDark] = useState(
    document.documentElement.classList.contains('dark')
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains('dark'));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  return isDark;
};

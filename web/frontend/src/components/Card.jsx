/**
 * Composants Card réutilisables avec effet Glassmorphism parfait
 *
 * Caractéristiques :
 * - Effet glassmorphism moderne avec backdrop-blur
 * - Bordures subtiles avec reflets lumineux
 * - Animations douces au hover
 * - Support mode sombre automatique via CSS variables
 */

export const Card = ({
  children,
  dark = false,
  glass = true,
  glow = false,
  hover = true,
  className = ''
}) => {
  const glassStyle = glass ? {
    background: 'rgba(var(--color-card-rgb), 0.5)',
    backdropFilter: 'blur(24px) saturate(180%)',
    WebkitBackdropFilter: 'blur(24px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    boxShadow: glow
      ? '0 8px 40px rgba(var(--color-primary-rgb), 0.15), 0 4px 16px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.4)'
      : '0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.4)',
  } : {
    backgroundColor: 'var(--color-card)',
    border: '1px solid var(--color-border)'
  };

  return (
    <div
      className={`
        rounded-[20px] p-6
        ${hover ? 'glass-hover' : ''}
        ${className}
      `}
      style={{
        ...glassStyle,
        color: 'var(--color-text)',
        transition: 'all 0.35s ease'
      }}
    >
      {children}
    </div>
  );
};

export const CardHeader = ({
  icon: Icon,
  title,
  subtitle,
  iconColor = 'text-[var(--color-primary)]',
  iconBgColor = '',
  className = ''
}) => {
  return (
    <div className={`mb-6 flex items-center gap-4 ${className}`}>
      {Icon && (
        <span
          className={`flex h-11 w-11 items-center justify-center rounded-2xl ${iconColor} glass-sm`}
          style={{
            background: 'linear-gradient(135deg, rgba(var(--color-primary-rgb), 0.15), rgba(var(--color-primary-rgb), 0.05))',
            border: '1px solid rgba(var(--color-primary-rgb), 0.2)',
            boxShadow: '0 4px 12px rgba(var(--color-primary-rgb), 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
          }}
        >
          <Icon size={20} />
        </span>
      )}
      <div>
        <h2
          className="text-lg font-semibold"
          style={{ color: 'var(--color-text)' }}
        >
          {title}
        </h2>
        {subtitle && (
          <p
            className="text-xs uppercase tracking-[0.32em]"
            style={{ color: 'var(--color-muted)' }}
          >
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
};

/**
 * Exemple d'utilisation :
 *
 * import { Card, CardHeader } from './components/Card';
 * import { TrendingUp } from 'lucide-react';
 *
 * <Card>
 *   <CardHeader
 *     icon={TrendingUp}
 *     title="Statistiques"
 *     subtitle="Vue d'ensemble"
 *   />
 *   <p>Contenu de la carte...</p>
 * </Card>
 */

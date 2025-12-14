/**
 * Composants Button réutilisables avec effet Glassmorphism parfait
 * 
 * Caractéristiques :
 * - Effet glassmorphism moderne avec backdrop-blur
 * - Animations fluides au hover et click
 * - Support mode sombre complet
 */

export const PrimaryButton = ({ 
  children, 
  onClick, 
  disabled = false,
  type = 'button',
  className = '' 
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        text-white 
        focus:ring-2 focus:outline-none
        disabled:opacity-50 disabled:cursor-not-allowed
        rounded-xl h-11 px-6 
        font-medium 
        transition-all duration-300
        ${className}
      `}
      style={{
        background: 'linear-gradient(135deg, rgba(var(--color-primary-rgb), 0.95), rgba(var(--color-primary-rgb), 0.8))',
        backdropFilter: 'blur(12px) saturate(180%)',
        WebkitBackdropFilter: 'blur(12px) saturate(180%)',
        border: '1px solid rgba(255, 255, 255, 0.25)',
        boxShadow: '0 8px 24px rgba(var(--color-primary-rgb), 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = 'linear-gradient(135deg, rgba(var(--color-primary-rgb), 1), rgba(var(--color-primary-rgb), 0.9))';
          e.currentTarget.style.boxShadow = '0 12px 32px rgba(var(--color-primary-rgb), 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }
      }}
      onMouseLeave={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = 'linear-gradient(135deg, rgba(var(--color-primary-rgb), 0.95), rgba(var(--color-primary-rgb), 0.8))';
          e.currentTarget.style.boxShadow = '0 8px 24px rgba(var(--color-primary-rgb), 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.3)';
          e.currentTarget.style.transform = 'translateY(0)';
        }
      }}
    >
      {children}
    </button>
  );
};

export const SecondaryButton = ({ 
  children, 
  onClick, 
  disabled = false,
  type = 'button',
  className = '' 
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        focus:ring-2 focus:outline-none
        disabled:opacity-50 disabled:cursor-not-allowed
        rounded-xl h-11 px-6 
        font-medium 
        transition-all duration-300
        ${className}
      `}
      style={{
        background: 'rgba(var(--color-card-rgb), 0.5)',
        backdropFilter: 'blur(12px) saturate(150%)',
        WebkitBackdropFilter: 'blur(12px) saturate(150%)',
        border: '1px solid rgba(var(--color-primary-rgb), 0.3)',
        color: 'var(--color-primary)',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.4)'
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = 'rgba(var(--color-card-rgb), 0.7)';
          e.currentTarget.style.borderColor = 'rgba(var(--color-primary-rgb), 0.5)';
          e.currentTarget.style.boxShadow = '0 8px 24px rgba(var(--color-primary-rgb), 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.5)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }
      }}
      onMouseLeave={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = 'rgba(var(--color-card-rgb), 0.5)';
          e.currentTarget.style.borderColor = 'rgba(var(--color-primary-rgb), 0.3)';
          e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.4)';
          e.currentTarget.style.transform = 'translateY(0)';
        }
      }}
    >
      {children}
    </button>
  );
};

export const GlassButton = ({ 
  children, 
  onClick, 
  disabled = false,
  type = 'button',
  className = '' 
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        focus:ring-2 focus:outline-none
        disabled:opacity-50 disabled:cursor-not-allowed
        rounded-xl h-11 px-6 
        font-medium 
        transition-all duration-300
        glass-button
        ${className}
      `}
      style={{
        color: 'var(--color-text)'
      }}
    >
      {children}
    </button>
  );
};

/**
 * Exemple d'utilisation :
 * 
 * import { PrimaryButton, SecondaryButton, GlassButton } from './components/Button';
 * 
 * <PrimaryButton onClick={handleSubmit}>
 *   Enregistrer
 * </PrimaryButton>
 * 
 * <SecondaryButton onClick={handleCancel}>
 *   Annuler
 * </SecondaryButton>
 * 
 * <GlassButton onClick={handleAction}>
 *   Action
 * </GlassButton>
 */

/**
 * AnimatedButton - Boutons avec animations avancées
 *
 * Caractéristiques :
 * - Effet ripple au clic
 * - Animations hover fluides
 * - Variantes : primary, secondary, ghost, gradient
 * - Loading state avec spinner
 * - Support des icônes
 */

import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

const buttonVariants = {
  initial: { scale: 1 },
  hover: {
    scale: 1.02,
    transition: { type: "spring", stiffness: 400, damping: 17 }
  },
  tap: {
    scale: 0.98,
    transition: { type: "spring", stiffness: 400, damping: 17 }
  }
};

const shimmerVariants = {
  initial: { x: '-100%' },
  hover: {
    x: '100%',
    transition: { duration: 0.6, ease: "easeInOut" }
  }
};

export const AnimatedButton = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  icon: Icon,
  iconPosition = 'left',
  className = '',
  onClick,
  ...props
}) => {
  const sizeClasses = {
    sm: 'h-9 px-4 text-sm',
    md: 'h-11 px-6 text-base',
    lg: 'h-14 px-8 text-lg'
  };

  const variantStyles = {
    primary: {
      background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-hover))',
      color: 'white',
      border: 'none',
      boxShadow: '0 4px 15px rgba(157, 54, 86, 0.3)'
    },
    secondary: {
      background: 'transparent',
      color: 'var(--color-primary)',
      border: '2px solid var(--color-primary)'
    },
    ghost: {
      background: 'transparent',
      color: 'var(--color-text)',
      border: '1px solid var(--color-border)'
    },
    gradient: {
      background: 'linear-gradient(135deg, #9D3656, #C86D8A, #F5C3CE)',
      backgroundSize: '200% 200%',
      color: 'white',
      border: 'none',
      boxShadow: '0 4px 20px rgba(157, 54, 86, 0.4)'
    },
    success: {
      background: 'linear-gradient(135deg, #2ED573, #1ABC9C)',
      color: 'white',
      border: 'none',
      boxShadow: '0 4px 15px rgba(46, 213, 115, 0.3)'
    },
    danger: {
      background: 'linear-gradient(135deg, #DC2626, #EF4444)',
      color: 'white',
      border: 'none',
      boxShadow: '0 4px 15px rgba(220, 38, 38, 0.3)'
    }
  };

  return (
    <motion.button
      className={`
        relative overflow-hidden
        font-semibold rounded-xl
        inline-flex items-center justify-center gap-2
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)]
        disabled:opacity-50 disabled:cursor-not-allowed
        transition-shadow
        ${sizeClasses[size]}
        ${className}
      `}
      style={variantStyles[variant]}
      variants={buttonVariants}
      initial="initial"
      whileHover={!disabled && !loading ? "hover" : undefined}
      whileTap={!disabled && !loading ? "tap" : undefined}
      onClick={onClick}
      disabled={disabled || loading}
      {...props}
    >
      {/* Shimmer effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
        variants={shimmerVariants}
        initial="initial"
        whileHover="hover"
      />

      {/* Content */}
      <span className="relative z-10 flex items-center gap-2">
        {loading ? (
          <Loader2 className="animate-spin" size={size === 'sm' ? 16 : size === 'lg' ? 24 : 20} />
        ) : (
          <>
            {Icon && iconPosition === 'left' && <Icon size={size === 'sm' ? 16 : size === 'lg' ? 24 : 20} />}
            {children}
            {Icon && iconPosition === 'right' && <Icon size={size === 'sm' ? 16 : size === 'lg' ? 24 : 20} />}
          </>
        )}
      </span>
    </motion.button>
  );
};

/**
 * IconButton - Bouton icône avec animation
 */
export const IconButton = ({
  icon: Icon,
  variant = 'ghost',
  size = 'md',
  className = '',
  onClick,
  tooltip,
  ...props
}) => {
  const sizeClasses = {
    sm: 'h-8 w-8',
    md: 'h-10 w-10',
    lg: 'h-12 w-12'
  };

  const iconSizes = {
    sm: 16,
    md: 20,
    lg: 24
  };

  return (
    <motion.button
      className={`
        relative overflow-hidden
        rounded-xl
        inline-flex items-center justify-center
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)]
        transition-colors
        ${sizeClasses[size]}
        ${className}
      `}
      style={{
        background: variant === 'ghost' ? 'transparent' : 'var(--color-secondary)',
        color: 'var(--color-primary)',
        border: '1px solid var(--color-border)'
      }}
      whileHover={{
        scale: 1.1,
        backgroundColor: 'var(--color-secondary)'
      }}
      onClick={onClick}
      title={tooltip}
      {...props}
    >
      <motion.div
        whileHover={{ rotate: 15 }}
        transition={{ type: "spring", stiffness: 400 }}
      >
        <Icon size={iconSizes[size]} />
      </motion.div>
    </motion.button>
  );
};

/**
 * FloatingActionButton - Bouton flottant animé
 */
export const FloatingActionButton = ({
  icon: Icon,
  onClick,
  className = '',
  ...props
}) => {
  return (
    <motion.button
      className={`
        fixed bottom-8 right-8
        h-14 w-14 rounded-full
        flex items-center justify-center
        shadow-lg
        focus:outline-none
        z-50
        ${className}
      `}
      style={{
        background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-light))',
        boxShadow: '0 8px 25px rgba(157, 54, 86, 0.4)'
      }}
      initial={{ scale: 0, rotate: -180 }}
      animate={{ scale: 1, rotate: 0 }}
      whileHover={{
        scale: 1.1,
        boxShadow: '0 12px 35px rgba(157, 54, 86, 0.5)'
      }}
      onClick={onClick}
      {...props}
    >
      <Icon size={24} className="text-white" />
    </motion.button>
  );
};

export default AnimatedButton;

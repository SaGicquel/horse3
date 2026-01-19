/**
 * GlassCard - Composant Card avec effet glassmorphism parfait
 *
 * Caractéristiques :
 * - Effet glassmorphism moderne avec backdrop-blur et saturate
 * - Bordures subtiles avec reflets lumineux
 * - Animations d'entrée et hover fluides
 * - Support des thèmes clair/sombre
 */

import { motion } from 'framer-motion';

const glassVariants = {
  hidden: {
    opacity: 0,
    y: 20,
    scale: 0.98
  },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
      mass: 1
    }
  },
  hover: {
    y: -4,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 25
    }
  }
};

export const GlassCard = ({
  children,
  className = '',
  animate = true,
  hover = true,
  glow = false,
  gradient = false,
  shimmer = false,
  delay = 0,
  onClick
}) => {
  const cardClasses = `p-6 glass-card ${shimmer ? 'glass-shimmer' : ''} ${gradient ? 'glass-border-glow' : ''} ${glow ? 'shadow-lg shadow-[#ec489933]' : ''} text-neutral-900 dark:text-neutral-100 ${className}`;

  if (!animate) {
    return (
      <div
        className={cardClasses}
        onClick={onClick}
      >
        {children}
      </div>
    );
  }

  return (
    <motion.div
      className={cardClasses}
      variants={glassVariants}
      initial="hidden"
      animate="visible"
      whileHover={hover ? "hover" : undefined}
      transition={{ delay }}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
};

/**
 * GlassCardHeader - En-tête pour GlassCard avec glassmorphism
 */
export const GlassCardHeader = ({
  icon: Icon,
  title,
  subtitle,
  action
}) => {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-4">
        {Icon && (
          <motion.div
            className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-[#ec4899] to-[#db2777] shadow-lg shadow-[#ec48994d] border border-white/20"
          >
            <Icon size={22} className="text-white" />
          </motion.div>
        )}
        <div>
          <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            {title}
          </h2>
          {subtitle && (
            <p className="text-xs uppercase tracking-wider text-neutral-600 dark:text-neutral-400">
              {subtitle}
            </p>
          )}
        </div>
      </div>
      {action && (
        <div>{action}</div>
      )}
    </div>
  );
};

export default GlassCard;

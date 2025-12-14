/**
 * PageTransition - Wrapper pour transitions de page fluides
 * 
 * Utilise Framer Motion AnimatePresence pour des transitions
 * entre les pages avec différents effets disponibles.
 */

import { motion, AnimatePresence } from 'framer-motion';

// Variantes de transition disponibles
const transitionVariants = {
  fade: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 }
  },
  slideUp: {
    initial: { opacity: 0, y: 40 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -40 }
  },
  slideDown: {
    initial: { opacity: 0, y: -40 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 40 }
  },
  slideLeft: {
    initial: { opacity: 0, x: 40 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -40 }
  },
  slideRight: {
    initial: { opacity: 0, x: -40 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 40 }
  },
  scale: {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 1.1 }
  },
  scaleRotate: {
    initial: { opacity: 0, scale: 0.9, rotate: -5 },
    animate: { opacity: 1, scale: 1, rotate: 0 },
    exit: { opacity: 0, scale: 0.9, rotate: 5 }
  },
  blur: {
    initial: { opacity: 0, filter: 'blur(10px)' },
    animate: { opacity: 1, filter: 'blur(0px)' },
    exit: { opacity: 0, filter: 'blur(10px)' }
  }
};

const defaultTransition = {
  type: "spring",
  stiffness: 100,
  damping: 20,
  mass: 1
};

/**
 * PageTransition Component
 * 
 * @param {string} variant - Type de transition (fade, slideUp, slideDown, etc.)
 * @param {string} pageKey - Clé unique pour la page (utilisée pour AnimatePresence)
 * @param {number} duration - Durée de la transition
 * @param {ReactNode} children - Contenu de la page
 */
export const PageTransition = ({ 
  children, 
  pageKey,
  variant = 'slideUp',
  duration = 0.5,
  className = ''
}) => {
  const variants = transitionVariants[variant] || transitionVariants.slideUp;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={pageKey}
        initial="initial"
        animate="animate"
        exit="exit"
        variants={variants}
        transition={{
          ...defaultTransition,
          duration
        }}
        className={className}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
};

/**
 * StaggerContainer - Container pour animations stagger
 */
export const StaggerContainer = ({ 
  children, 
  staggerDelay = 0.1,
  className = '' 
}) => {
  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: { opacity: 0 },
        visible: {
          opacity: 1,
          transition: {
            staggerChildren: staggerDelay,
            delayChildren: 0.1
          }
        }
      }}
    >
      {children}
    </motion.div>
  );
};

/**
 * StaggerItem - Élément enfant pour StaggerContainer
 */
export const StaggerItem = ({ 
  children, 
  className = '',
  variant = 'fadeUp'
}) => {
  const itemVariants = {
    fadeUp: {
      hidden: { opacity: 0, y: 20 },
      visible: { 
        opacity: 1, 
        y: 0,
        transition: { type: "spring", stiffness: 100, damping: 15 }
      }
    },
    fadeIn: {
      hidden: { opacity: 0 },
      visible: { 
        opacity: 1,
        transition: { duration: 0.5 }
      }
    },
    scaleIn: {
      hidden: { opacity: 0, scale: 0.8 },
      visible: { 
        opacity: 1, 
        scale: 1,
        transition: { type: "spring", stiffness: 200, damping: 20 }
      }
    },
    slideIn: {
      hidden: { opacity: 0, x: -20 },
      visible: { 
        opacity: 1, 
        x: 0,
        transition: { type: "spring", stiffness: 100, damping: 15 }
      }
    }
  };

  return (
    <motion.div
      className={className}
      variants={itemVariants[variant] || itemVariants.fadeUp}
    >
      {children}
    </motion.div>
  );
};

/**
 * ScrollReveal - Animation au scroll
 */
export const ScrollReveal = ({ 
  children, 
  className = '',
  direction = 'up',
  delay = 0,
  duration = 0.6
}) => {
  const directions = {
    up: { y: 60, x: 0 },
    down: { y: -60, x: 0 },
    left: { x: 60, y: 0 },
    right: { x: -60, y: 0 }
  };

  return (
    <motion.div
      className={className}
      initial={{ 
        opacity: 0, 
        ...directions[direction]
      }}
      whileInView={{ 
        opacity: 1, 
        y: 0, 
        x: 0 
      }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{
        duration,
        delay,
        ease: [0.25, 0.1, 0.25, 1]
      }}
    >
      {children}
    </motion.div>
  );
};

/**
 * ParallaxSection - Section avec effet parallax
 */
export const ParallaxSection = ({ 
  children, 
  className = '',
  speed = 0.5 
}) => {
  return (
    <motion.div
      className={className}
      initial={{ y: 0 }}
      whileInView={{ y: 0 }}
      viewport={{ once: false }}
      style={{
        y: 0
      }}
      transition={{
        type: "tween",
        ease: "linear"
      }}
    >
      {children}
    </motion.div>
  );
};

/**
 * HoverCard - Card avec effet 3D au hover
 */
export const HoverCard = ({ 
  children, 
  className = '',
  intensity = 10
}) => {
  return (
    <motion.div
      className={`${className} perspective-1000`}
      whileHover={{
        rotateX: intensity / 2,
        rotateY: intensity,
        transition: { type: "spring", stiffness: 300, damping: 20 }
      }}
      style={{ transformStyle: 'preserve-3d' }}
    >
      {children}
    </motion.div>
  );
};

export default PageTransition;

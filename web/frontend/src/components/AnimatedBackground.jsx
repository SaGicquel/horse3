/**
 * AnimatedBackground - Backgrounds animés modernes
 * 
 * Différents effets de fond :
 * - Gradient mesh animé
 * - Particules flottantes
 * - Grille perspective
 * - Blobs morphing
 */

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

/**
 * GradientMesh - Fond avec gradient mesh animé
 */
export const GradientMesh = ({ className = '' }) => {
  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`}>
      {/* Gradient principal */}
      <div 
        className="absolute inset-0"
        style={{
          background: 'var(--color-bg)'
        }}
      />
      
      {/* Blob 1 */}
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full opacity-30"
        style={{
          background: 'radial-gradient(circle, rgba(157, 54, 86, 0.4) 0%, transparent 70%)',
          top: '-10%',
          left: '-10%',
          filter: 'blur(60px)'
        }}
        animate={{
          x: [0, 100, 50, 0],
          y: [0, 50, 100, 0],
          scale: [1, 1.2, 1.1, 1]
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      {/* Blob 2 */}
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full opacity-25"
        style={{
          background: 'radial-gradient(circle, rgba(200, 109, 138, 0.4) 0%, transparent 70%)',
          bottom: '10%',
          right: '-5%',
          filter: 'blur(50px)'
        }}
        animate={{
          x: [0, -80, -40, 0],
          y: [0, -60, -30, 0],
          scale: [1, 1.1, 1.2, 1]
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      {/* Blob 3 */}
      <motion.div
        className="absolute w-[400px] h-[400px] rounded-full opacity-20"
        style={{
          background: 'radial-gradient(circle, rgba(245, 195, 206, 0.5) 0%, transparent 70%)',
          top: '40%',
          left: '30%',
          filter: 'blur(40px)'
        }}
        animate={{
          x: [0, 60, -30, 0],
          y: [0, -40, 60, 0],
          scale: [1, 1.15, 0.95, 1]
        }}
        transition={{
          duration: 18,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
    </div>
  );
};

/**
 * FloatingParticles - Particules flottantes
 */
export const FloatingParticles = ({ 
  count = 20, 
  className = '' 
}) => {
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    const newParticles = Array.from({ length: count }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      duration: Math.random() * 20 + 10,
      delay: Math.random() * 5
    }));
    setParticles(newParticles);
  }, [count]);

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            width: particle.size,
            height: particle.size,
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            background: 'var(--color-primary)',
            opacity: 0.3
          }}
          animate={{
            y: [0, -100, 0],
            x: [0, Math.random() * 50 - 25, 0],
            opacity: [0.3, 0.6, 0.3],
            scale: [1, 1.5, 1]
          }}
          transition={{
            duration: particle.duration,
            delay: particle.delay,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      ))}
    </div>
  );
};

/**
 * GridBackground - Grille avec effet perspective
 */
export const GridBackground = ({ className = '' }) => {
  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`}>
      <div 
        className="absolute inset-0"
        style={{
          background: 'var(--color-bg)'
        }}
      />
      <div 
        className="absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(to right, var(--color-border) 1px, transparent 1px),
            linear-gradient(to bottom, var(--color-border) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
          maskImage: 'radial-gradient(ellipse at center, black 30%, transparent 80%)'
        }}
      />
      {/* Ligne horizon animée */}
      <motion.div
        className="absolute left-0 right-0 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent, var(--color-primary), transparent)',
          top: '50%'
        }}
        animate={{
          opacity: [0.3, 0.6, 0.3],
          scaleX: [0.8, 1, 0.8]
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
    </div>
  );
};

/**
 * MorphingBlobs - Blobs avec effet morphing
 */
export const MorphingBlobs = ({ className = '' }) => {
  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`}>
      <motion.div
        className="absolute w-96 h-96"
        style={{
          background: 'linear-gradient(135deg, rgba(157, 54, 86, 0.3), rgba(200, 109, 138, 0.2))',
          filter: 'blur(40px)',
          top: '10%',
          left: '10%'
        }}
        animate={{
          borderRadius: [
            '60% 40% 30% 70% / 60% 30% 70% 40%',
            '30% 60% 70% 40% / 50% 60% 30% 60%',
            '50% 60% 30% 60% / 30% 60% 70% 40%',
            '60% 40% 30% 70% / 60% 30% 70% 40%'
          ]
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      <motion.div
        className="absolute w-72 h-72"
        style={{
          background: 'linear-gradient(135deg, rgba(245, 195, 206, 0.3), rgba(157, 54, 86, 0.2))',
          filter: 'blur(30px)',
          bottom: '15%',
          right: '15%'
        }}
        animate={{
          borderRadius: [
            '30% 60% 70% 40% / 50% 60% 30% 60%',
            '60% 40% 30% 70% / 60% 30% 70% 40%',
            '50% 60% 30% 60% / 30% 60% 70% 40%',
            '30% 60% 70% 40% / 50% 60% 30% 60%'
          ]
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
    </div>
  );
};

/**
 * NeonGlow - Effet néon avec lignes animées
 */
export const NeonGlow = ({ className = '' }) => {
  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {/* Ligne horizontale */}
      <motion.div
        className="absolute h-px w-full"
        style={{
          background: 'linear-gradient(90deg, transparent, var(--color-primary), transparent)',
          top: '30%',
          boxShadow: '0 0 20px var(--color-primary), 0 0 40px var(--color-primary)'
        }}
        animate={{
          opacity: [0, 0.8, 0],
          x: ['-100%', '100%']
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
          repeatDelay: 2
        }}
      />
      
      {/* Ligne verticale */}
      <motion.div
        className="absolute w-px h-full"
        style={{
          background: 'linear-gradient(180deg, transparent, var(--color-primary-light), transparent)',
          left: '70%',
          boxShadow: '0 0 15px var(--color-primary-light)'
        }}
        animate={{
          opacity: [0, 0.6, 0],
          y: ['-100%', '100%']
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
          repeatDelay: 1
        }}
      />
    </div>
  );
};

/**
 * CombinedBackground - Combinaison des effets
 */
export const CombinedBackground = ({ 
  variant = 'default',
  className = '' 
}) => {
  const variants = {
    default: (
      <>
        <GradientMesh />
        <FloatingParticles count={15} />
      </>
    ),
    grid: (
      <>
        <GridBackground />
        <NeonGlow />
      </>
    ),
    blobs: (
      <>
        <MorphingBlobs />
        <FloatingParticles count={10} />
      </>
    ),
    minimal: (
      <GradientMesh />
    )
  };

  return (
    <div className={`fixed inset-0 -z-10 ${className}`}>
      {variants[variant] || variants.default}
    </div>
  );
};

export default CombinedBackground;

/**
 * Skeleton - Composants de chargement modernes avec Glassmorphism
 * 
 * Différents types de skeletons pour un chargement élégant :
 * - SkeletonCard
 * - SkeletonText
 * - SkeletonAvatar
 * - SkeletonChart
 * - SkeletonTable
 */

import { motion } from 'framer-motion';

// Animation de base pour le shimmer
const shimmerVariants = {
  animate: {
    backgroundPosition: ['200% 0', '-200% 0'],
    transition: {
      duration: 2,
      ease: "linear",
      repeat: Infinity
    }
  }
};

// Composant de base Skeleton avec glassmorphism
export const Skeleton = ({
  width = '100%',
  height = 20,
  borderRadius = 8,
  className = ''
}) => {
  return (
    <motion.div
      className={`relative overflow-hidden bg-neutral-200/50 dark:bg-neutral-800/50 backdrop-blur-sm ${className}`}
      style={{
        width,
        height,
        borderRadius,
      }}
    >
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
        variants={shimmerVariants}
        animate="animate"
        style={{ backgroundSize: '200% 100%' }}
      />
    </motion.div>
  );
};

// Skeleton pour texte
export const SkeletonText = ({
  lines = 3,
  className = ''
}) => {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          width={i === lines - 1 ? '60%' : '100%'}
          height={16}
        />
      ))}
    </div>
  );
};

// Skeleton pour avatar
export const SkeletonAvatar = ({
  size = 48,
  className = ''
}) => {
  return (
    <Skeleton
      width={size}
      height={size}
      borderRadius="50%"
      className={className}
    />
  );
};

// Skeleton pour carte de statistiques avec glassmorphism
export const SkeletonStatCard = ({ className = '' }) => {
  return (
    <motion.div
      className={`rounded-[20px] p-6 glass-card ${className}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Skeleton width={44} height={44} borderRadius={12} />
          <Skeleton width={60} height={24} borderRadius={20} />
        </div>
      </div>
      <Skeleton width={120} height={40} className="mb-2" />
      <Skeleton width={80} height={16} />
    </motion.div>
  );
};

// Skeleton pour graphique
export const SkeletonChart = ({
  height = 300,
  className = ''
}) => {
  return (
    <motion.div
      className={`rounded-2xl p-6 glass-card ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="flex items-center gap-4 mb-6">
        <Skeleton width={44} height={44} borderRadius={12} />
        <div className="space-y-2">
          <Skeleton width={150} height={20} />
          <Skeleton width={100} height={14} />
        </div>
      </div>
      <div style={{ height }} className="relative">
        {/* Barres simulées */}
        <div className="absolute bottom-0 left-0 right-0 flex items-end justify-around gap-2 h-full">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton
              key={i}
              width="10%"
              height={`${Math.random() * 60 + 20}%`}
              borderRadius={8}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
};

// Skeleton pour tableau
export const SkeletonTable = ({
  rows = 5,
  columns = 4,
  className = ''
}) => {
  return (
    <motion.div
      className={`rounded-2xl overflow-hidden glass-card ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex gap-4 p-4 border-b border-white/10">
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton key={i} width={`${100 / columns}%`} height={20} />
        ))}
      </div>

      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div
          key={rowIndex}
          className={`flex gap-4 p-4 ${rowIndex < rows - 1 ? 'border-b border-white/10' : ''}`}
        >
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton
              key={colIndex}
              width={`${100 / columns}%`}
              height={16}
            />
          ))}
        </div>
      ))}
    </motion.div>
  );
};

// Skeleton pour liste de performances
export const SkeletonPerformanceList = ({
  items = 4,
  className = ''
}) => {
  return (
    <motion.div
      className={`rounded-2xl p-6 glass-card ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="flex items-center gap-4 mb-6">
        <Skeleton width={44} height={44} borderRadius={12} />
        <Skeleton width={180} height={24} />
      </div>

      <div className="space-y-4">
        {Array.from({ length: items }).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-4 p-3 rounded-lg bg-neutral-50/50 dark:bg-neutral-900/50"
          >
            <div className="flex-1 space-y-2">
              <Skeleton width="70%" height={16} />
              <Skeleton width="100%" height={8} borderRadius={4} />
            </div>
            <Skeleton width={60} height={28} borderRadius={6} />
          </div>
        ))}
      </div>
    </motion.div>
  );
};

// Skeleton pour la page Dashboard complète
export const SkeletonDashboard = () => {
  return (
    <div className="max-w-7xl mx-auto px-3 sm:px-6 py-6 sm:py-12 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <Skeleton width={200} height={32} />
        <Skeleton width={300} height={16} />
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonStatCard key={i} />
        ))}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2">
          <SkeletonPerformanceList />
        </div>
        <div className="lg:col-span-3">
          <SkeletonChart />
        </div>
      </div>
    </div>
  );
};

export default Skeleton;

/**
 * AnimatedStatCard - Carte de statistiques avec animations avancées
 * 
 * Caractéristiques :
 * - Animation du compteur numérique
 * - Effet glassmorphism
 * - Icône animée
 * - Progress ring optionnel
 * - Sparkline mini-graphique
 */

import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

// Hook pour animer les nombres
const useAnimatedNumber = (value, duration = 1) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const numValue = parseFloat(value?.toString().replace(/[^0-9.-]/g, '')) || 0;

    const controls = animate(0, numValue, {
      duration,
      ease: "easeOut",
      onUpdate: (latest) => {
        setDisplayValue(latest);
      }
    });

    return controls.stop;
  }, [value, duration]);

  return displayValue;
};

// Composant Progress Ring
const ProgressRing = ({ progress = 0, size = 60, strokeWidth = 4 }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle
        className="stroke-neutral-200 dark:stroke-neutral-700"
        fill="transparent"
        strokeWidth={strokeWidth}
        r={radius}
        cx={size / 2}
        cy={size / 2}
      />
      <motion.circle
        className="stroke-primary-500"
        fill="transparent"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        r={radius}
        cx={size / 2}
        cy={size / 2}
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1.5, ease: "easeOut" }}
        style={{
          strokeDasharray: circumference
        }}
      />
    </svg>
  );
};

// Mini Sparkline
const Sparkline = ({ data = [], color = 'var(--color-primary)' }) => {
  if (!data.length) return null;

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - ((value - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <motion.svg
      viewBox="0 0 100 100"
      className="w-24 h-8"
      preserveAspectRatio="none"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.5 }}
    >
      <motion.polyline
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      />
    </motion.svg>
  );
};

const AnimatedStatCard = ({
  title,
  value,
  suffix = '',
  prefix = '',
  change,
  changeLabel,
  icon: Icon,
  trend = 'up',
  progress,
  sparklineData,
  delay = 0
}) => {
  const isPositive = trend === 'up';
  const animatedValue = useAnimatedNumber(value, 1.5);

  // Formatage du nombre
  const formatValue = (num) => {
    if (suffix === '%') return num.toFixed(1);
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return Math.round(num).toLocaleString('fr-FR');
  };

  return (
    <motion.div
      className="relative overflow-hidden rounded-2xl p-6 h-full min-h-[160px] flex flex-col justify-between glass-card"
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        delay,
        type: "spring",
        stiffness: 100,
        damping: 15
      }}
      whileHover={{
        y: -5,
      }}
    >
      {/* Background gradient blob */}
      <motion.div
        className={`absolute -top-10 -right-10 w-32 h-32 rounded-full opacity-20 ${isPositive ? 'bg-primary-500' : 'bg-error'
          }`}
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.2, 0.3, 0.2]
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />

      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <motion.div
            className="flex items-center gap-3"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: delay + 0.2 }}
          >
            {Icon && (
              <motion.div
                className={`p-2.5 rounded-xl border ${isPositive
                    ? 'bg-primary-500/10 border-primary-500/20 text-primary-600 dark:text-primary-400'
                    : 'bg-error/10 border-error/20 text-error'
                  }`}
                whileHover={{ scale: 1.1, rotate: 5 }}
              >
                <Icon size={20} />
              </motion.div>
            )}
            {change && (
              <motion.div
                className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold ${isPositive
                    ? 'bg-success/10 text-success'
                    : 'bg-error/10 text-error'
                  }`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: delay + 0.4, type: "spring" }}
              >
                {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                {change}
              </motion.div>
            )}
          </motion.div>

          {/* Progress Ring ou Sparkline */}
          {progress !== undefined && (
            <ProgressRing progress={progress} />
          )}
          {sparklineData && (
            <Sparkline data={sparklineData} />
          )}
        </div>

        {/* Value */}
        <motion.div
          className="mb-2"
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: delay + 0.3, type: "spring", stiffness: 200 }}
        >
          <span className="text-4xl font-bold tracking-tight text-neutral-900 dark:text-neutral-100">
            {prefix}{formatValue(animatedValue)}{suffix}
          </span>
        </motion.div>

        {/* Title */}
        <motion.p
          className="text-sm font-medium text-neutral-500 dark:text-neutral-400"
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: delay + 0.4 }}
        >
          {title}
        </motion.p>

        {changeLabel && (
          <motion.p
            className="text-xs mt-1 text-neutral-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: delay + 0.5 }}
          >
            {changeLabel}
          </motion.p>
        )}
      </div>
    </motion.div>
  );
};

export default AnimatedStatCard;

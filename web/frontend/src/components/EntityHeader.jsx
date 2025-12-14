import { motion } from 'framer-motion';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';
import GlassCard from './GlassCard';

const EntityHeader = ({
  name,
  emoji,
  subtitle,
  pill,
  metrics = [],
  trendData = [],
  onCta,
  ctaLabel = 'Voir courses liées'
}) => {
  const safeTrend = trendData?.length ? trendData : [{ label: 'N/A', value: 0 }];

  return (
    <GlassCard className="p-4 sm:p-6 border border-white/10">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div className="flex items-center gap-3">
            {emoji && <span className="text-3xl">{emoji}</span>}
            <div>
              {pill && (
                <span className="inline-flex items-center text-[10px] uppercase tracking-widest px-2 py-1 rounded-full bg-white/10 border border-white/10 text-gray-300">
                  {pill}
                </span>
              )}
              <h2 className="text-2xl sm:text-3xl font-bold text-[var(--color-text)] leading-tight">
                {name || '--'}
              </h2>
              {subtitle && <p className="text-gray-400 text-sm mt-1">{subtitle}</p>}
            </div>
          </div>

          {onCta && (
            <motion.button
              onClick={onCta}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium shadow-lg shadow-purple-500/10"
              style={{ 
                background: 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))',
                color: '#0b1021'
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <ArrowRightIcon className="h-4 w-4" />
              {ctaLabel}
            </motion.button>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {metrics.map((metric) => (
            <div key={metric.label} className="rounded-lg p-3 bg-white/5 border border-white/10">
              <p className="text-[11px] uppercase text-gray-400 tracking-wide">{metric.label}</p>
              <p className="text-xl font-semibold text-[var(--color-text)] mt-1">
                {metric.value}
              </p>
              {metric.hint && <p className="text-xs text-gray-500 mt-0.5">{metric.hint}</p>}
            </div>
          ))}
        </div>

        <div className="h-24 sm:h-28">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={safeTrend}>
              <Line type="monotone" dataKey="value" stroke="var(--color-primary)" strokeWidth={2} dot={false} />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: 'var(--color-bg, #0b1021)', 
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px',
                  color: 'var(--color-text)'
                }}
                labelFormatter={(label) => label || 'Date'}
                formatter={(val) => [`${val}%`, 'Taux de réussite']}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-1">Tendance 30 derniers jours</p>
        </div>
      </div>
    </GlassCard>
  );
};

export default EntityHeader;

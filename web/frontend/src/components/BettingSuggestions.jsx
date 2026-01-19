import React from 'react';
import { motion } from 'framer-motion';
import { DollarSign, TrendingUp, AlertCircle } from 'lucide-react';
import { GlassCard } from './GlassCard';

const BettingSuggestions = ({ suggestions, bankrollStats }) => {
  if (!suggestions || suggestions.length === 0) {
    return (
      <div className="p-6 text-center text-neutral-400">
        <AlertCircle className="mx-auto mb-2 opacity-50" />
        <p>Aucun pari suggéré pour cette course (Value insuffisante ou risque trop élevé)</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-sm font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
          Stratégie de Mise (Kelly)
        </h4>
        <span className="text-xs px-2 py-1 rounded bg-neutral-100 dark:bg-white/5 text-neutral-500">
          Bankroll: {bankrollStats?.total || 1000}€
        </span>
      </div>

      {suggestions.map((bet, index) => (
        <GlassCard
          key={bet.cheval_id}
          className="relative overflow-hidden !p-4 !bg-white/50 dark:!bg-black/20"
          hover={false}
          animate={true}
          delay={index * 0.1}
        >
          <div className="flex justify-between items-start mb-2 relative z-10">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-neutral-900 dark:text-white">
                  #{bet.numero} {bet.nom}
                </span>
                <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded border border-primary/20">
                  WIN
                </span>
              </div>
              <div className="text-xs text-neutral-500 mt-1 flex gap-2">
                <span>Edge: <span className="text-success font-bold">+{bet.value_edge * 100}%</span></span>
                <span>Kelly: {bet.kelly_fraction}</span>
              </div>
            </div>

            <div className="text-right">
              <div className="text-xl font-mono font-bold text-primary flex items-center justify-end">
                {bet.mise_conseillee}€
              </div>
              <div className="text-xs text-neutral-400">
                {bet.pourcentage_bankroll}% bankroll
              </div>
            </div>
          </div>

          {/* Progress bar background */}
          <div className="absolute bottom-0 left-0 w-full h-1 bg-neutral-100 dark:bg-white/5">
            <motion.div
              className="h-full bg-gradient-to-r from-primary to-secondary"
              initial={{ width: 0 }}
              animate={{ width: `${(bet.mise_conseillee / (bankrollStats?.max_stake || 50)) * 100}%` }}
              transition={{ duration: 1, delay: 0.2 }}
            />
          </div>
        </GlassCard>
      ))}
    </div>
  );
};

export default BettingSuggestions;

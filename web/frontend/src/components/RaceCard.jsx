import React from 'react';
import { motion } from 'framer-motion';
import { Clock, MapPin, Trophy, Users, AlertTriangle } from 'lucide-react';
import { GlassCard } from './GlassCard';

const RaceCard = ({ race, onAnalyze, isAnalyzing }) => {
  return (
    <GlassCard hover={true} className="flex flex-col h-full relative overflow-hidden group">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-bold text-neutral-900 dark:text-white flex items-center gap-2">
            <span className="bg-primary/10 text-primary px-2 py-0.5 rounded text-sm">R{race.reunion}C{race.course}</span>
            {race.nom || `Course ${race.course}`}
          </h3>
          <div className="flex items-center gap-2 text-sm text-neutral-500 dark:text-neutral-400 mt-1">
            <MapPin size={14} />
            <span>{race.hippodrome}</span>
          </div>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-lg font-mono font-bold text-primary">
            {race.heure_depart}
          </span>
          <span className="text-xs text-neutral-400 bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 rounded-full border border-neutral-200 dark:border-neutral-700">
            {race.statut || 'À venir'}
          </span>
        </div>
      </div>

      {/* Details Grid */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        <div className="flex items-center gap-2 bg-neutral-50 dark:bg-white/5 p-2 rounded-lg">
          <Trophy size={16} className="text-yellow-500" />
          <div className="flex flex-col">
            <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Discipline</span>
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-200">{race.discipline}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-neutral-50 dark:bg-white/5 p-2 rounded-lg">
          <Users size={16} className="text-blue-500" />
          <div className="flex flex-col">
            <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Partants</span>
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-200">{race.partants?.length || 0}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-neutral-50 dark:bg-white/5 p-2 rounded-lg col-span-2">
          <Clock size={16} className="text-green-500" />
          <div className="flex flex-col">
            <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Distance</span>
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-200">{race.distance}m</span>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="mt-auto">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onAnalyze(race)}
          disabled={isAnalyzing}
          className={`w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${
            isAnalyzing
              ? 'bg-neutral-100 text-neutral-400 cursor-not-allowed dark:bg-white/5'
              : 'bg-gradient-to-r from-primary to-secondary text-white shadow-lg shadow-primary/25 hover:shadow-primary/40'
          }`}
        >
          {isAnalyzing ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Analyse en cours...</span>
            </>
          ) : (
            <>
              <span className="text-lg">✨</span>
              <span>Lancer l'Analyse IA</span>
            </>
          )}
        </motion.button>
      </div>

      {/* Decorative gradient blob */}
      <div className="absolute -top-10 -right-10 w-32 h-32 bg-primary/5 rounded-full blur-3xl pointer-events-none group-hover:bg-primary/10 transition-colors duration-500" />
    </GlassCard>
  );
};

export default RaceCard;

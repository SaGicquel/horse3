import React from 'react';
import { motion } from 'framer-motion';
import { BeakerIcon, PlayIcon } from '@heroicons/react/24/outline';

/**
 * Toggle pour basculer entre Mode Réel et Mode Simulation
 * Stocke l'état dans localStorage pour persistance
 */
export function SimulationToggle({ isSimulation, onToggle }) {
    return (
        <div className="flex items-center gap-3 p-2 rounded-xl bg-slate-800/50 border border-slate-700/50">
            {/* Label Mode Réel */}
            <button
                onClick={() => onToggle(false)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${!isSimulation
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'text-slate-400 hover:text-slate-300'
                    }`}
            >
                <PlayIcon className="w-4 h-4" />
                <span className="text-sm font-medium">Réel</span>
            </button>

            {/* Label Mode Simulation */}
            <button
                onClick={() => onToggle(true)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${isSimulation
                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                        : 'text-slate-400 hover:text-slate-300'
                    }`}
            >
                <BeakerIcon className="w-4 h-4" />
                <span className="text-sm font-medium">Simulation</span>
            </button>
        </div>
    );
}

/**
 * Badge pour indiquer qu'on est en mode simulation
 */
export function SimulationBadge({ className = '' }) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/20 border border-amber-500/30 ${className}`}
        >
            <BeakerIcon className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-xs font-medium text-amber-400">SIMULATION</span>
        </motion.div>
    );
}

/**
 * Hook pour gérer l'état simulation avec localStorage
 */
export function useSimulationMode() {
    const [isSimulation, setIsSimulation] = React.useState(() => {
        const stored = localStorage.getItem('simulation_mode');
        return stored === 'true';
    });

    const toggleSimulation = React.useCallback((value) => {
        const newValue = typeof value === 'boolean' ? value : !isSimulation;
        setIsSimulation(newValue);
        localStorage.setItem('simulation_mode', String(newValue));
    }, [isSimulation]);

    return { isSimulation, toggleSimulation };
}

export default SimulationToggle;

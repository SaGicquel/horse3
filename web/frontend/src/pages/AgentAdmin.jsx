/**
 * 🤖 Agent Admin - Dashboard Complet avec Mémoire & Backtesting
 * ==============================================================
 *
 * Interface pour:
 * - Visualiser les runs Agent IA
 * - Voir les performances (accuracy, PnL)
 * - Consulter les leçons apprises
 * - Lancer des backtests
 */

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ArrowPathIcon,
    CheckCircleIcon,
    XCircleIcon,
    ClockIcon,
    ChevronDownIcon,
    ChevronUpIcon,
    CpuChipIcon,
    BanknotesIcon,
    SparklesIcon,
    ExclamationTriangleIcon,
    ChartBarIcon,
    AcademicCapIcon,
    BeakerIcon,
    Cog6ToothIcon,
    PlayIcon,
} from '@heroicons/react/24/outline';

const API_BASE = '';

// =============================================================================
// STATUS BADGE COMPONENT
// =============================================================================

const StatusBadge = ({ status }) => {
    const configs = {
        SUCCESS: { bg: 'bg-green-500/20', text: 'text-green-400', icon: CheckCircleIcon },
        WIN: { bg: 'bg-green-500/20', text: 'text-green-400', icon: CheckCircleIcon },
        FAILED: { bg: 'bg-red-500/20', text: 'text-red-400', icon: XCircleIcon },
        LOSE: { bg: 'bg-red-500/20', text: 'text-red-400', icon: XCircleIcon },
        RUNNING: { bg: 'bg-blue-500/20', text: 'text-blue-400', icon: ArrowPathIcon },
        PENDING: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: ClockIcon },
        STEP_A: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: ClockIcon },
        STEP_B: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: ClockIcon },
        STEP_C: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: ClockIcon },
        STEP_D: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: ClockIcon },
    };
    const config = configs[status] || { bg: 'bg-neutral-500/20', text: 'text-neutral-400', icon: ClockIcon };
    const Icon = config.icon;

    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
            <Icon className={`w-3.5 h-3.5 ${status === 'RUNNING' ? 'animate-spin' : ''}`} />
            {status}
        </span>
    );
};

// =============================================================================
// TAB NAVIGATION
// =============================================================================

const TabButton = ({ active, onClick, icon: Icon, label }) => (
    <button
        onClick={onClick}
        className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all ${active
            ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
            : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
            }`}
    >
        <Icon className="w-4 h-4" />
        {label}
    </button>
);

// =============================================================================
// STAT CARD
// =============================================================================

const StatCard = ({ title, value, subtitle, color = 'primary', icon: Icon }) => (
    <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-500">{title}</span>
            {Icon && <Icon className={`w-4 h-4 text-${color}-400`} />}
        </div>
        <div className={`text-2xl font-bold text-${color}-400`}>{value}</div>
        {subtitle && <div className="text-xs text-neutral-500 mt-1">{subtitle}</div>}
    </div>
);

// =============================================================================
// PERFORMANCE TAB
// =============================================================================

const PerformanceTab = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [syncing, setSyncing] = useState(false);

    const fetchStats = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/agent/stats`);
            const data = await res.json();
            if (data.success) {
                setStats(data);
            }
        } catch (e) {
            console.error('Error fetching stats:', e);
        }
        setLoading(false);
    }, []);

    const syncOutcomes = async () => {
        setSyncing(true);
        try {
            const res = await fetch(`${API_BASE}/agent/outcomes/sync`, { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                fetchStats();
            }
        } catch (e) {
            console.error('Error syncing:', e);
        }
        setSyncing(false);
    };

    useEffect(() => {
        fetchStats();
    }, [fetchStats]);

    if (loading) {
        return (
            <div className="flex items-center justify-center p-12">
                <ArrowPathIcon className="w-8 h-8 text-primary-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Actions */}
            <div className="flex justify-end gap-3">
                <button
                    onClick={syncOutcomes}
                    disabled={syncing}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm transition-colors disabled:opacity-50"
                >
                    <ArrowPathIcon className={`w-4 h-4 ${syncing ? 'animate-spin' : ''}`} />
                    Synchroniser résultats
                </button>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                    title="Total Prédictions"
                    value={stats?.total_predictions || 0}
                    icon={SparklesIcon}
                />
                <StatCard
                    title="Win Rate"
                    value={`${stats?.win_rate || 0}%`}
                    color={stats?.win_rate >= 50 ? 'green' : 'red'}
                    icon={ChartBarIcon}
                />
                <StatCard
                    title="PnL Total"
                    value={`${stats?.total_pnl >= 0 ? '+' : ''}${stats?.total_pnl?.toFixed(2) || 0}€`}
                    color={stats?.total_pnl >= 0 ? 'green' : 'red'}
                    icon={BanknotesIcon}
                />
                <StatCard
                    title="Confiance Moyenne"
                    value={`${stats?.avg_confidence || 0}%`}
                    icon={CpuChipIcon}
                />
            </div>

            {/* Daily Performance */}
            {stats?.daily_performance?.length > 0 && (
                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4">
                    <h3 className="text-lg font-semibold text-white mb-4">Performance Journalière</h3>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead className="bg-neutral-800/50">
                                <tr>
                                    <th className="text-left text-xs font-medium text-neutral-400 px-3 py-2">Date</th>
                                    <th className="text-left text-xs font-medium text-neutral-400 px-3 py-2">Paris</th>
                                    <th className="text-left text-xs font-medium text-neutral-400 px-3 py-2">Wins</th>
                                    <th className="text-left text-xs font-medium text-neutral-400 px-3 py-2">PnL</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-neutral-800">
                                {stats.daily_performance.slice(0, 10).map((day, i) => (
                                    <tr key={i} className="hover:bg-neutral-800/30">
                                        <td className="px-3 py-2 text-white">{day.date}</td>
                                        <td className="px-3 py-2 text-neutral-300">{day.bets}</td>
                                        <td className="px-3 py-2 text-green-400">{day.wins}</td>
                                        <td className={`px-3 py-2 ${day.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                            {day.pnl >= 0 ? '+' : ''}{day.pnl?.toFixed(2)}€
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* By Bet Type */}
            {stats?.by_bet_type?.length > 0 && (
                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4">
                    <h3 className="text-lg font-semibold text-white mb-4">Par Type de Pari</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {stats.by_bet_type.map((bt, i) => (
                            <div key={i} className="bg-neutral-800/50 rounded-lg p-3">
                                <div className="font-medium text-white mb-1">{bt.bet_type}</div>
                                <div className="flex items-center gap-4 text-sm">
                                    <span className="text-neutral-400">{bt.total} paris</span>
                                    <span className="text-green-400">{bt.win_rate}% win</span>
                                    <span className={bt.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                                        {bt.pnl >= 0 ? '+' : ''}{bt.pnl?.toFixed(2)}€
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!stats?.total_predictions && (
                <div className="text-center py-12 text-neutral-500">
                    <ChartBarIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <div>Aucune donnée de performance</div>
                    <div className="text-sm mt-1">Lancez des analyses IA et synchronisez les résultats</div>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// LESSONS TAB
// =============================================================================

const LessonsTab = () => {
    const [lessons, setLessons] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchLessons = async () => {
            try {
                const res = await fetch(`${API_BASE}/agent/lessons`);
                const data = await res.json();
                if (data.success) {
                    setLessons(data.lessons);
                }
            } catch (e) {
                console.error('Error fetching lessons:', e);
            }
            setLoading(false);
        };
        fetchLessons();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center p-12">
                <ArrowPathIcon className="w-8 h-8 text-primary-500 animate-spin" />
            </div>
        );
    }

    const tierColors = {
        EXCELLENT: 'border-green-500/50 bg-green-500/10',
        GOOD: 'border-blue-500/50 bg-blue-500/10',
        AVERAGE: 'border-yellow-500/50 bg-yellow-500/10',
        POOR: 'border-red-500/50 bg-red-500/10',
    };

    return (
        <div className="space-y-4">
            {lessons.length > 0 ? (
                lessons.map((lesson) => (
                    <div
                        key={lesson.id}
                        className={`border rounded-xl p-4 ${tierColors[lesson.performance_tier] || 'border-neutral-800 bg-neutral-900/50'}`}
                    >
                        <div className="flex items-start justify-between mb-2">
                            <div>
                                <span className="text-xs text-neutral-500">{lesson.lesson_type}</span>
                                <h4 className="font-medium text-white">{lesson.pattern_key}</h4>
                            </div>
                            <div className="text-right">
                                <div className={`text-lg font-bold ${lesson.accuracy_pct >= 50 ? 'text-green-400' : 'text-red-400'}`}>
                                    {lesson.accuracy_pct?.toFixed(1)}%
                                </div>
                                <div className="text-xs text-neutral-500">accuracy</div>
                            </div>
                        </div>
                        <p className="text-sm text-neutral-300">{lesson.lesson_text}</p>
                        <div className="flex items-center gap-4 mt-3 text-xs text-neutral-500">
                            <span>{lesson.total_predictions} prédictions</span>
                            <span>{lesson.correct_predictions} correctes</span>
                            <span className={lesson.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                                PnL: {lesson.total_pnl >= 0 ? '+' : ''}{lesson.total_pnl?.toFixed(2)}€
                            </span>
                        </div>
                    </div>
                ))
            ) : (
                <div className="text-center py-12 text-neutral-500">
                    <AcademicCapIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <div>Aucune leçon apprise</div>
                    <div className="text-sm mt-1">Les leçons apparaîtront après plusieurs prédictions synchronisées</div>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// BACKTEST TAB
// =============================================================================

const BacktestTab = () => {
    const [backtests, setBacktests] = useState([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [showForm, setShowForm] = useState(false);
    const [runningBacktest, setRunningBacktest] = useState(null);
    const [form, setForm] = useState({
        start_date: '',
        end_date: '',
        profile: 'STANDARD',
        bankroll: 500,
    });

    const fetchBacktests = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/agent/backtest/list`);
            const data = await res.json();
            if (data.success) {
                setBacktests(data.backtests);
                // Check if any backtest is running
                const running = data.backtests.find(bt => bt.status === 'RUNNING');
                setRunningBacktest(running || null);
            }
        } catch (e) {
            console.error('Error fetching backtests:', e);
        }
        setLoading(false);
    }, []);

    // Polling for running backtests
    useEffect(() => {
        let interval;
        if (runningBacktest) {
            interval = setInterval(async () => {
                try {
                    const res = await fetch(`${API_BASE}/agent/backtest/${runningBacktest.backtest_id}`);
                    const data = await res.json();
                    if (data.success) {
                        // Update the running backtest in the list
                        setBacktests(prev => prev.map(bt =>
                            bt.backtest_id === data.backtest_id ? data : bt
                        ));
                        // Check if still running
                        if (data.status !== 'RUNNING') {
                            setRunningBacktest(null);
                        } else {
                            setRunningBacktest(data);
                        }
                    }
                } catch (e) {
                    console.error('Error polling backtest:', e);
                }
            }, 2000); // Poll every 2 seconds
        }
        return () => clearInterval(interval);
    }, [runningBacktest]);

    useEffect(() => {
        fetchBacktests();
    }, [fetchBacktests]);

    const createBacktest = async () => {
        if (!form.start_date || !form.end_date) return;

        setCreating(true);
        try {
            const res = await fetch(`${API_BASE}/agent/backtest/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });
            const data = await res.json();
            if (data.success) {
                setShowForm(false);
                // Start polling for this backtest
                setRunningBacktest({ backtest_id: data.backtest_id, status: 'PENDING', progress_pct: 0 });
                fetchBacktests();
            }
        } catch (e) {
            console.error('Error creating backtest:', e);
        }
        setCreating(false);
    };

    // Progress Bar Component
    const ProgressBar = ({ progress, status, processingDate, totalDays, processedDays }) => (
        <div className="bg-neutral-900/50 border border-primary-500/30 rounded-xl p-4 mb-6">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <ArrowPathIcon className="w-5 h-5 text-primary-400 animate-spin" />
                    <span className="text-white font-medium">Backtest en cours...</span>
                </div>
                <span className="text-primary-400 font-bold text-lg">{progress}%</span>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-neutral-800 rounded-full h-3 overflow-hidden">
                <motion.div
                    className="h-full bg-gradient-to-r from-primary-500 to-primary-400 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                />
            </div>

            <div className="flex items-center justify-between mt-2 text-xs text-neutral-500">
                <span>
                    {processedDays || 0} / {totalDays || '?'} jours traités
                </span>
                {processingDate && (
                    <span>En cours: {processingDate}</span>
                )}
            </div>
        </div>
    );

    if (loading) {
        return (
            <div className="flex items-center justify-center p-12">
                <ArrowPathIcon className="w-8 h-8 text-primary-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Running Backtest Progress */}
            {runningBacktest && (runningBacktest.status === 'RUNNING' || runningBacktest.status === 'PENDING') && (
                <ProgressBar
                    progress={runningBacktest.progress_pct || 0}
                    status={runningBacktest.status}
                    processingDate={runningBacktest.processing_date}
                    totalDays={runningBacktest.total_days}
                    processedDays={runningBacktest.processed_days}
                />
            )}

            {/* New Backtest Button */}
            <div className="flex justify-end gap-3">
                <button
                    onClick={fetchBacktests}
                    className="flex items-center gap-2 px-3 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-sm transition-colors"
                >
                    <ArrowPathIcon className="w-4 h-4" />
                    Actualiser
                </button>
                <button
                    onClick={() => setShowForm(!showForm)}
                    disabled={runningBacktest?.status === 'RUNNING'}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm transition-colors disabled:opacity-50"
                >
                    <PlayIcon className="w-4 h-4" />
                    Nouveau Backtest
                </button>
            </div>

            {/* Form */}
            <AnimatePresence>
                {showForm && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
                    >
                        <h3 className="font-medium text-white mb-4">Créer un Backtest</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                                <label className="text-xs text-neutral-500 block mb-1">Date début</label>
                                <input
                                    type="date"
                                    value={form.start_date}
                                    onChange={(e) => setForm({ ...form, start_date: e.target.value })}
                                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white text-sm"
                                />
                            </div>
                            <div>
                                <label className="text-xs text-neutral-500 block mb-1">Date fin</label>
                                <input
                                    type="date"
                                    value={form.end_date}
                                    onChange={(e) => setForm({ ...form, end_date: e.target.value })}
                                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white text-sm"
                                />
                            </div>
                            <div>
                                <label className="text-xs text-neutral-500 block mb-1">Profil</label>
                                <select
                                    value={form.profile}
                                    onChange={(e) => setForm({ ...form, profile: e.target.value })}
                                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white text-sm"
                                >
                                    <option value="PRUDENT">Prudent</option>
                                    <option value="STANDARD">Standard</option>
                                    <option value="AGRESSIF">Agressif</option>
                                </select>
                            </div>
                            <div>
                                <label className="text-xs text-neutral-500 block mb-1">Bankroll (€)</label>
                                <input
                                    type="number"
                                    value={form.bankroll}
                                    onChange={(e) => setForm({ ...form, bankroll: parseFloat(e.target.value) })}
                                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white text-sm"
                                />
                            </div>
                        </div>
                        <div className="flex justify-end gap-3 mt-4">
                            <button
                                onClick={() => setShowForm(false)}
                                className="px-4 py-2 text-neutral-400 hover:text-white text-sm"
                            >
                                Annuler
                            </button>
                            <button
                                onClick={createBacktest}
                                disabled={creating || !form.start_date || !form.end_date}
                                className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm transition-colors disabled:opacity-50"
                            >
                                {creating && <ArrowPathIcon className="w-4 h-4 animate-spin" />}
                                Lancer
                            </button>
                        </div>
                        <p className="text-xs text-neutral-500 mt-2">
                            ⚠️ Limité à 7 jours maximum. Chaque jour fait un appel LLM.
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Backtests List */}
            {backtests.length > 0 ? (
                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
                    <table className="w-full">
                        <thead className="bg-neutral-800/50">
                            <tr>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Période</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Status</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Progression</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Accuracy</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">PnL</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-800">
                            {backtests.map((bt) => (
                                <tr key={bt.backtest_id} className="hover:bg-neutral-800/30">
                                    <td className="px-4 py-3 text-white">
                                        {bt.start_date} → {bt.end_date}
                                    </td>
                                    <td className="px-4 py-3">
                                        <StatusBadge status={bt.status} />
                                    </td>
                                    <td className="px-4 py-3">
                                        {bt.status === 'RUNNING' || bt.status === 'PENDING' ? (
                                            <div className="flex items-center gap-2">
                                                <div className="w-24 bg-neutral-700 rounded-full h-2 overflow-hidden">
                                                    <div
                                                        className="h-full bg-primary-500 rounded-full transition-all"
                                                        style={{ width: `${bt.progress_pct || 0}%` }}
                                                    />
                                                </div>
                                                <span className="text-xs text-primary-400">{bt.progress_pct || 0}%</span>
                                            </div>
                                        ) : (
                                            <span className="text-neutral-500 text-sm">
                                                {bt.processed_days || 0}/{bt.total_days || 0}
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-4 py-3 text-neutral-300">
                                        {bt.accuracy_pct ? `${bt.accuracy_pct}%` : '—'}
                                    </td>
                                    <td className={`px-4 py-3 ${(bt.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {bt.total_pnl ? `${bt.total_pnl >= 0 ? '+' : ''}${bt.total_pnl.toFixed(2)}€` : '—'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="text-center py-12 text-neutral-500">
                    <BeakerIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <div>Aucun backtest</div>
                    <div className="text-sm mt-1">Créez un backtest pour simuler l'agent sur des données passées</div>
                </div>
            )}
        </div>
    );
};


// =============================================================================
// RUNS TAB (existing functionality)
// =============================================================================

const StepCard = ({ step, isExpanded, onToggle }) => {
    const stepLabels = {
        A: { name: 'Rapport Algo', icon: '📊', color: 'from-blue-500 to-cyan-500' },
        B: { name: 'Analyse IA', icon: '🤖', color: 'from-purple-500 to-pink-500' },
        C: { name: 'Vérification', icon: '🔍', color: 'from-orange-500 to-amber-500' },
        D: { name: 'Proposition Finale', icon: '🎯', color: 'from-green-500 to-emerald-500' },
    };
    const label = stepLabels[step.step_name] || { name: step.step_name, icon: '⚙️', color: 'from-neutral-500 to-neutral-600' };

    return (
        <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
            <button
                onClick={onToggle}
                className="w-full flex items-center justify-between p-4 hover:bg-neutral-800/50 transition-colors"
            >
                <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${label.color} flex items-center justify-center text-lg`}>
                        {label.icon}
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Step {step.step_name}: {label.name}</div>
                        <div className="text-xs text-neutral-500">
                            {step.duration_ms ? `${(step.duration_ms / 1000).toFixed(1)}s` : '—'}
                            {step.llm_model && ` • ${step.llm_model}`}
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <StatusBadge status={step.status} />
                    {isExpanded ? <ChevronUpIcon className="w-5 h-5 text-neutral-500" /> : <ChevronDownIcon className="w-5 h-5 text-neutral-500" />}
                </div>
            </button>
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-neutral-800"
                    >
                        <div className="p-4 space-y-3">
                            {step.error_message && (
                                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
                                    ❌ {step.error_message}
                                </div>
                            )}
                            {step.output_json && (
                                <div>
                                    <div className="text-xs text-neutral-500 mb-1">Output:</div>
                                    <pre className="bg-neutral-950 rounded-lg p-3 text-xs text-neutral-300 overflow-x-auto max-h-48 overflow-y-auto">
                                        {JSON.stringify(step.output_json, null, 2)}
                                    </pre>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

const RunDetail = ({ runId, onBack }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [expandedSteps, setExpandedSteps] = useState({});

    useEffect(() => {
        const fetchDetail = async () => {
            try {
                const res = await fetch(`${API_BASE}/agent/runs/${runId}`);
                const json = await res.json();
                if (json.run) {
                    setData(json);
                    const expanded = {};
                    (json.steps || []).forEach(s => {
                        if (s.status === 'FAILED') expanded[s.step_id] = true;
                    });
                    setExpandedSteps(expanded);
                }
            } catch (e) {
                console.error('Error:', e);
            }
            setLoading(false);
        };
        fetchDetail();
    }, [runId]);

    if (loading) return <div className="flex justify-center p-12"><ArrowPathIcon className="w-8 h-8 text-primary-500 animate-spin" /></div>;
    if (!data) return <div className="text-red-400 text-center">Run non trouvé</div>;

    const { run, steps } = data;

    return (
        <div className="space-y-6">
            <button onClick={onBack} className="text-neutral-400 hover:text-white">← Retour</button>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard title="Date" value={run.target_date} />
                <StatCard title="Durée" value={run.duration_ms ? `${(run.duration_ms / 1000).toFixed(1)}s` : '—'} />
                <StatCard title="Picks" value={run.total_picks_final || 0} />
                <StatCard title="Confiance" value={run.confidence_score ? `${run.confidence_score}%` : '—'} />
            </div>
            <div className="space-y-3">
                {steps.map(step => (
                    <StepCard
                        key={step.step_id}
                        step={step}
                        isExpanded={expandedSteps[step.step_id]}
                        onToggle={() => setExpandedSteps(prev => ({ ...prev, [step.step_id]: !prev[step.step_id] }))}
                    />
                ))}
            </div>
        </div>
    );
};

const RunsTab = () => {
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedRunId, setSelectedRunId] = useState(null);

    const fetchRuns = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/agent/runs?limit=20`);
            const json = await res.json();
            if (json.runs) setRuns(json.runs);
        } catch (e) {
            console.error('Error:', e);
        }
        setLoading(false);
    }, []);

    useEffect(() => {
        fetchRuns();
    }, [fetchRuns]);

    if (selectedRunId) {
        return <RunDetail runId={selectedRunId} onBack={() => setSelectedRunId(null)} />;
    }

    if (loading) return <div className="flex justify-center p-12"><ArrowPathIcon className="w-8 h-8 text-primary-500 animate-spin" /></div>;

    return (
        <div className="space-y-4">
            <div className="flex justify-end">
                <button onClick={fetchRuns} className="flex items-center gap-2 px-3 py-1.5 text-sm bg-neutral-800 hover:bg-neutral-700 rounded-lg text-neutral-300">
                    <ArrowPathIcon className="w-4 h-4" /> Actualiser
                </button>
            </div>
            {runs.length > 0 ? (
                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
                    <table className="w-full">
                        <thead className="bg-neutral-800/50">
                            <tr>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Date</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Status</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Durée</th>
                                <th className="text-left text-xs font-medium text-neutral-400 px-4 py-3">Picks</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-800">
                            {runs.map(run => (
                                <tr
                                    key={run.run_id}
                                    onClick={() => setSelectedRunId(run.run_id)}
                                    className="hover:bg-neutral-800/50 cursor-pointer"
                                >
                                    <td className="px-4 py-3 text-white">{run.target_date}</td>
                                    <td className="px-4 py-3"><StatusBadge status={run.status} /></td>
                                    <td className="px-4 py-3 text-neutral-300">{run.duration_ms ? `${(run.duration_ms / 1000).toFixed(1)}s` : '—'}</td>
                                    <td className="px-4 py-3 text-neutral-300">{run.total_picks_final || 0}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="text-center py-12 text-neutral-500">
                    <CpuChipIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <div>Aucun run Agent IA</div>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// TRAINING TAB - Automated Training with Learning Curve
// =============================================================================

const TrainingTab = () => {
    const [sessions, setSessions] = useState([]);
    const [currentSession, setCurrentSession] = useState(null);
    const [loading, setLoading] = useState(true);
    const [starting, setStarting] = useState(false);
    const [startDate, setStartDate] = useState('2020-01-01');
    const [endDate, setEndDate] = useState('2024-12-31');
    const [periodType, setPeriodType] = useState('WEEK');

    const fetchSessions = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/agent/training/list`);
            const json = await res.json();
            if (json.sessions) {
                setSessions(json.sessions);
                // Auto-select running/paused session
                const active = json.sessions.find(s => s.status === 'RUNNING' || s.status === 'PAUSED');
                if (active) {
                    fetchSessionDetail(active.session_id);
                }
            }
        } catch (e) {
            console.error(e);
        }
        setLoading(false);
    }, []);

    const fetchSessionDetail = async (sessionId) => {
        try {
            const res = await fetch(`${API_BASE}/agent/training/${sessionId}`);
            const json = await res.json();
            if (json.success) {
                setCurrentSession(json);
            }
        } catch (e) {
            console.error(e);
        }
    };

    useEffect(() => {
        fetchSessions();
    }, [fetchSessions]);

    // Poll for updates when session is running
    useEffect(() => {
        if (currentSession?.status === 'RUNNING') {
            const interval = setInterval(() => {
                fetchSessionDetail(currentSession.session_id);
            }, 5000);
            return () => clearInterval(interval);
        }
    }, [currentSession?.status, currentSession?.session_id]);

    const startTraining = async () => {
        setStarting(true);
        try {
            // Utilise le mode LLM avec vrai Gemini
            const res = await fetch(`${API_BASE}/agent/training/start-llm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start_date: startDate, end_date: endDate, period_type: 'DAY' })
            });
            const json = await res.json();
            if (json.success) {
                fetchSessions();
                fetchSessionDetail(json.session_id);
            }
        } catch (e) {
            console.error(e);
        }
        setStarting(false);
    };

    const pauseTraining = async () => {
        if (!currentSession) return;
        await fetch(`${API_BASE}/agent/training/${currentSession.session_id}/pause`, { method: 'POST' });
        fetchSessionDetail(currentSession.session_id);
    };

    const resumeTraining = async () => {
        if (!currentSession) return;
        await fetch(`${API_BASE}/agent/training/${currentSession.session_id}/resume`, { method: 'POST' });
        fetchSessionDetail(currentSession.session_id);
    };

    if (loading) {
        return <div className="flex items-center justify-center p-12">
            <ArrowPathIcon className="w-8 h-8 text-primary-500 animate-spin" />
        </div>;
    }

    return (
        <div className="space-y-6">
            {/* New Training Form */}
            <div className="bg-neutral-900/50 rounded-xl border border-neutral-800 p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <AcademicCapIcon className="w-5 h-5 text-primary-400" />
                    Nouvel Entraînement
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label className="block text-sm text-neutral-400 mb-1">Date début</label>
                        <input
                            type="date"
                            value={startDate}
                            onChange={e => setStartDate(e.target.value)}
                            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-neutral-400 mb-1">Date fin</label>
                        <input
                            type="date"
                            value={endDate}
                            onChange={e => setEndDate(e.target.value)}
                            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-neutral-400 mb-1">Période</label>
                        <select
                            value={periodType}
                            onChange={e => setPeriodType(e.target.value)}
                            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white"
                        >
                            <option value="WEEK">Semaine</option>
                            <option value="MONTH">Mois</option>
                        </select>
                    </div>
                    <div className="flex items-end">
                        <button
                            onClick={startTraining}
                            disabled={starting || currentSession?.status === 'RUNNING'}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50"
                        >
                            <PlayIcon className="w-4 h-4" />
                            {starting ? 'Démarrage...' : 'Démarrer'}
                        </button>
                    </div>
                </div>
            </div>

            {/* Current Session */}
            {currentSession && (
                <div className="bg-neutral-900/50 rounded-xl border border-neutral-800 p-6">
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                Session Active
                                <StatusBadge status={currentSession.status} />
                            </h3>
                            <p className="text-sm text-neutral-400">
                                {currentSession.start_date} → {currentSession.end_date}
                            </p>
                        </div>
                        <div className="flex gap-2">
                            {currentSession.status === 'RUNNING' && (
                                <button onClick={pauseTraining} className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm">
                                    ⏸️ Pause
                                </button>
                            )}
                            {currentSession.status === 'PAUSED' && (
                                <button onClick={resumeTraining} className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm">
                                    ▶️ Reprendre
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="mb-6">
                        <div className="flex justify-between text-sm text-neutral-400 mb-2">
                            <span>Progression</span>
                            <span>{currentSession.periods_completed} / {currentSession.total_periods} périodes ({currentSession.progress_pct}%)</span>
                        </div>
                        <div className="w-full bg-neutral-800 rounded-full h-3">
                            <div
                                className="bg-primary-500 h-3 rounded-full transition-all duration-500"
                                style={{ width: `${currentSession.progress_pct}%` }}
                            />
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-4 gap-4 mb-6">
                        <div className="bg-neutral-800/50 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-white">{currentSession.total_predictions}</div>
                            <div className="text-xs text-neutral-400">Prédictions</div>
                        </div>
                        <div className="bg-neutral-800/50 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-green-400">{currentSession.accuracy_pct}%</div>
                            <div className="text-xs text-neutral-400">Accuracy</div>
                        </div>
                        <div className={`bg-neutral-800/50 rounded-lg p-3 text-center`}>
                            <div className={`text-2xl font-bold ${currentSession.cumulative_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {currentSession.cumulative_pnl >= 0 ? '+' : ''}{currentSession.cumulative_pnl.toFixed(0)}€
                            </div>
                            <div className="text-xs text-neutral-400">PnL Cumulé</div>
                        </div>
                        <div className="bg-neutral-800/50 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-purple-400">{currentSession.lessons_generated}</div>
                            <div className="text-xs text-neutral-400">Leçons</div>
                        </div>
                    </div>

                    {/* Learning Curve */}
                    {currentSession.learning_curve && currentSession.learning_curve.length > 0 && (
                        <div>
                            <h4 className="text-sm font-medium text-white mb-3">📈 Courbe d'Apprentissage</h4>
                            <div className="bg-neutral-800/50 rounded-lg p-4 overflow-x-auto">
                                <div className="flex items-end gap-1 h-32 min-w-max">
                                    {currentSession.learning_curve.map((point, i) => (
                                        <div key={i} className="flex flex-col items-center group relative">
                                            <div
                                                className="w-4 bg-primary-500 rounded-t transition-all hover:bg-primary-400"
                                                style={{ height: `${point.accuracy * 1.2}px` }}
                                                title={`${point.start}: ${point.accuracy}%`}
                                            />
                                            {i % 4 === 0 && (
                                                <div className="text-[9px] text-neutral-500 mt-1 -rotate-45 origin-left whitespace-nowrap">
                                                    {point.start?.slice(5) || ''}
                                                </div>
                                            )}
                                            {/* Tooltip on hover */}
                                            <div className="absolute bottom-full mb-2 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                                                {point.accuracy}% | {point.predictions} paris | {point.pnl}€
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Sessions History */}
            <div className="bg-neutral-900/50 rounded-xl border border-neutral-800 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Historique des Sessions</h3>
                {sessions.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-neutral-800">
                                    <th className="px-4 py-2 text-left text-xs font-medium text-neutral-500">Période</th>
                                    <th className="px-4 py-2 text-left text-xs font-medium text-neutral-500">Status</th>
                                    <th className="px-4 py-2 text-left text-xs font-medium text-neutral-500">Progress</th>
                                    <th className="px-4 py-2 text-left text-xs font-medium text-neutral-500">Prédictions</th>
                                    <th className="px-4 py-2 text-left text-xs font-medium text-neutral-500">Accuracy</th>
                                    <th className="px-4 py-2 text-left text-xs font-medium text-neutral-500">Leçons</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-neutral-800">
                                {sessions.map(s => (
                                    <tr
                                        key={s.session_id}
                                        className="hover:bg-neutral-800/50 cursor-pointer"
                                        onClick={() => fetchSessionDetail(s.session_id)}
                                    >
                                        <td className="px-4 py-3 text-neutral-300 text-sm">{s.start_date} → {s.end_date}</td>
                                        <td className="px-4 py-3"><StatusBadge status={s.status} /></td>
                                        <td className="px-4 py-3 text-neutral-300 text-sm">{s.progress_pct}%</td>
                                        <td className="px-4 py-3 text-neutral-300 text-sm">{s.total_predictions}</td>
                                        <td className="px-4 py-3 text-green-400 text-sm">{s.accuracy_pct}%</td>
                                        <td className="px-4 py-3 text-purple-400 text-sm">{s.lessons_generated}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="text-center py-8 text-neutral-500">
                        Aucune session d'entraînement
                    </div>
                )}
            </div>
        </div>
    );
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

const AgentAdmin = () => {
    const [activeTab, setActiveTab] = useState('performance');

    const tabs = [
        { id: 'performance', label: 'Performance', icon: ChartBarIcon },
        { id: 'runs', label: 'Runs', icon: CpuChipIcon },
        { id: 'lessons', label: 'Leçons', icon: AcademicCapIcon },
        { id: 'backtest', label: 'Backtest', icon: BeakerIcon },
        { id: 'training', label: '🎓 Entraînement', icon: SparklesIcon },
    ];

    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                    <CpuChipIcon className="w-7 h-7 text-primary-400" />
                    Agent IA - Administration
                </h1>
                <p className="text-neutral-400 mt-1">
                    Performance, leçons et backtesting de l'Agent IA
                </p>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mb-6 flex-wrap">
                {tabs.map(tab => (
                    <TabButton
                        key={tab.id}
                        active={activeTab === tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        icon={tab.icon}
                        label={tab.label}
                    />
                ))}
            </div>

            {/* Content */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.15 }}
                >
                    {activeTab === 'performance' && <PerformanceTab />}
                    {activeTab === 'runs' && <RunsTab />}
                    {activeTab === 'lessons' && <LessonsTab />}
                    {activeTab === 'backtest' && <BacktestTab />}
                    {activeTab === 'training' && <TrainingTab />}
                </motion.div>
            </AnimatePresence>
        </div>
    );
};

export default AgentAdmin;

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Info,
  Loader2,
  RefreshCw,
  Settings2,
  Sparkles,
  Target
} from 'lucide-react';
import { GlassCard } from '../components/GlassCard';
import { API_BASE } from '../config/api';
import { Skeleton } from '../components/Skeleton';

const mockHealth = {
  ece: 0.018,
  brier: 0.082,
  last_calibration: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString()
};

const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return `${(value * 100).toFixed(2)}%`;
};

const formatRelativeTime = (dateString) => {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return 'N/A';

  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMinutes < 1) return "A l'instant";
  if (diffMinutes < 60) return `Il y a ${diffMinutes} min`;
  if (diffHours < 24) return `Il y a ${diffHours}h`;
  if (diffDays < 7) return `Il y a ${diffDays} j`;

  return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
};

const formatFullDate = (dateString) => {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return 'N/A';
  return date.toLocaleString('fr-FR', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const toastTone = {
  success: {
    className: 'glass-notification-success',
    icon: CheckCircle2,
    color: 'text-green-500 dark:text-green-400'
  },
  warning: {
    className: 'glass-notification-warning',
    icon: AlertTriangle,
    color: 'text-amber-500 dark:text-amber-400'
  },
  info: {
    className: 'glass-notification-info',
    icon: Info,
    color: 'text-blue-500 dark:text-blue-400'
  },
  error: {
    className: 'glass-notification-error',
    icon: AlertTriangle,
    color: 'text-red-500 dark:text-red-400'
  }
};

export default function Calibration() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [actionLoading, setActionLoading] = useState({ recalibrate: false, sync: false });
  const [isMocked, setIsMocked] = useState(false);
  const [toasts, setToasts] = useState([]);
  const toastTimeouts = useRef([]);
  const mockNotifiedRef = useRef(false);

  const pushToast = useCallback((type, message) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setToasts((prev) => [...prev, { id, type, message }]);

    const timeoutId = window.setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
      toastTimeouts.current = toastTimeouts.current.filter((t) => t !== timeoutId);
    }, 4200);

    toastTimeouts.current.push(timeoutId);
  }, []);

  useEffect(() => {
    return () => {
      toastTimeouts.current.forEach((timeoutId) => clearTimeout(timeoutId));
    };
  }, []);

  const loadHealth = useCallback(
    async ({ asRefresh = false, notify = false } = {}) => {
      if (asRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      let usedMock = false;

      try {
        const response = await fetch(`${API_BASE}/calibration/health`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        setHealth(data);
        setIsMocked(false);

        if (notify) {
          pushToast('success', 'Métriques de calibration rafraîchies');
        }
      } catch (error) {
        console.error('Erreur chargement calibration:', error);
        usedMock = true;
        setHealth(mockHealth);
        setIsMocked(true);

        if (!mockNotifiedRef.current || !asRefresh) {
          pushToast('info', 'Endpoints calibration indisponibles, affichage de données mock.');
          mockNotifiedRef.current = true;
        }
      } finally {
        if (asRefresh) {
          setRefreshing(false);
        } else {
          setLoading(false);
        }
      }

      return usedMock;
    },
    [pushToast]
  );

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  const handleAction = useCallback(
    async (action) => {
      setActionLoading((prev) => ({ ...prev, [action]: true }));
      const label = action === 'recalibrate' ? 'Recalibrage' : 'Sync config';
      let simulated = false;

      try {
        const endpoint = action === 'recalibrate' ? '/calibration/recalibrate' : '/calibration/sync_config';
        const response = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        pushToast('success', `${label} lancé`);
      } catch (error) {
        console.error(`Erreur action ${action}:`, error);
        simulated = true;
        pushToast('warning', `${label} simulé (endpoint indisponible)`);
      } finally {
        setActionLoading((prev) => ({ ...prev, [action]: false }));
        const usedMock = await loadHealth({ asRefresh: true, notify: !simulated });
        if (simulated && usedMock) {
          pushToast('info', 'Métriques rafraîchies en mode mock.');
        }
      }
    },
    [loadHealth, pushToast]
  );

  const warnings = useMemo(() => {
    if (!health) return [];

    const ece = health.ece_7d ?? health.ece;
    const alerts = [];

    if (ece !== undefined && ece > 0.02) {
      alerts.push(`ECE élevé: ${formatPercent(ece)} (cible < 2%)`);
    }

    if (health.last_calibration) {
      const lastDate = new Date(health.last_calibration);
      const daysSince = Math.floor((Date.now() - lastDate.getTime()) / (1000 * 60 * 60 * 24));
      if (daysSince > 7) {
        alerts.push(`Calibration obsolète (${daysSince} jours)`);
      }
    }

    return alerts;
  }, [health]);

  const renderMetricCard = (title, value, hint, accent, Icon) => (
    <GlassCard key={title} hover className="h-full p-4">
      <div className="flex items-start gap-3">
        <div
          className="w-11 h-11 rounded-xl flex items-center justify-center bg-primary-100 dark:bg-primary-900/20"
        >
          <Icon size={18} className={accent} />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-sm text-neutral-600 dark:text-neutral-400">{title}</span>
          <span className={`text-3xl font-bold ${accent}`}>{value}</span>
          <span className="text-xs text-neutral-500 dark:text-neutral-500">{hint}</span>
        </div>
      </div>
    </GlassCard>
  );

  const eceValue = health?.ece_7d ?? health?.ece;
  const brierValue = health?.brier_7d ?? health?.brier;
  const metricCards = [
    {
      title: 'ECE (Expected Calibration Error)',
      value: formatPercent(eceValue),
      hint: 'Cible < 2% sur 7j',
      accent: 'text-emerald-600 dark:text-emerald-400',
      icon: Target
    },
    {
      title: 'Brier score',
      value: formatPercent(brierValue),
      hint: 'Plus bas = mieux',
      accent: 'text-blue-600 dark:text-blue-400',
      icon: Activity
    },
    {
      title: 'Derniere calibration',
      value: formatRelativeTime(health?.last_calibration),
      hint: formatFullDate(health?.last_calibration),
      accent: 'text-purple-600 dark:text-purple-400',
      icon: Clock3
    }
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-6 px-4 sm:px-0">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
      >
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <Sparkles className="text-emerald-500 dark:text-emerald-400" size={18} />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 dark:from-green-400 dark:to-emerald-400 bg-clip-text text-transparent">
              Calibration
            </h1>
          </div>
          <p className="text-neutral-600 dark:text-neutral-400">
            Santé des probabilités, écart aux résultats et suivi des recalibrages.
          </p>
          {isMocked && (
            <div className="inline-flex items-center gap-2 text-xs px-3 py-1 rounded-full bg-amber-100 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-400/30 text-amber-800 dark:text-amber-300 w-fit">
              <AlertTriangle size={14} /> Mode mock (API en attente)
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => loadHealth({ asRefresh: true, notify: true })}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-50 bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-white/10 transition-colors"
          >
            <motion.div
              animate={refreshing ? { rotate: 360 } : {}}
              transition={{ duration: 1, repeat: refreshing ? Infinity : 0, ease: 'linear' }}
            >
              {refreshing ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
            </motion.div>
            Rafraichir
          </button>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {loading
          ? [1, 2, 3].map((idx) => (
            <GlassCard key={idx} animate={false} className="p-4">
              <div className="flex flex-col gap-3">
                <Skeleton width="40%" height={16} />
                <Skeleton width="70%" height={34} />
                <Skeleton width="60%" height={14} />
              </div>
            </GlassCard>
          ))
          : metricCards.map((card) => renderMetricCard(card.title, card.value, card.hint, card.accent, card.icon))}
      </div>

      <GlassCard className="p-6">
        <div className="flex flex-col gap-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Actions calibration</h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">Déclencher un recalibrage ou synchroniser la config modèle.</p>
            </div>
            <div className="flex gap-3 flex-wrap">
              <button
                onClick={() => handleAction('recalibrate')}
                disabled={actionLoading.recalibrate}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold text-white disabled:opacity-60 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 border border-blue-500/20 shadow-lg shadow-blue-500/20"
              >
                {actionLoading.recalibrate ? <Loader2 size={16} className="animate-spin" /> : <Target size={16} />}
                Recalibrer
              </button>
              <button
                onClick={() => handleAction('sync')}
                disabled={actionLoading.sync}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-60 bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-white/10 transition-colors"
              >
                {actionLoading.sync ? <Loader2 size={16} className="animate-spin" /> : <Settings2 size={16} />}
                Sync config
              </button>
            </div>
          </div>

          {warnings.length > 0 && (
            <div
              className="p-4 rounded-xl border flex flex-col gap-2 bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-400/30"
            >
              <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
                <AlertTriangle size={18} />
                <span className="text-sm font-semibold">A surveiller</span>
              </div>
              <ul className="text-sm text-amber-800 dark:text-amber-200 space-y-1 list-disc list-inside">
                {warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              className="p-4 rounded-xl flex flex-col gap-2 bg-neutral-50 dark:bg-white/5 border border-neutral-200 dark:border-white/10"
            >
              <span className="text-xs uppercase tracking-wide text-neutral-500 dark:text-neutral-400">Etat pipeline</span>
              <div className="flex items-center gap-2 text-sm">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 dark:bg-emerald-400 shadow-[0_0_0_4px_rgba(16,185,129,0.2)]" />
                <span className="text-neutral-900 dark:text-neutral-100">
                  {health ? 'Calibration chargée' : 'Lecture en cours'}
                  {isMocked ? ' (mock)' : ''}
                </span>
              </div>
              <span className="text-xs text-neutral-500 dark:text-neutral-400">
                Dernière sync: {formatRelativeTime(health?.last_calibration)} — {formatFullDate(health?.last_calibration)}
              </span>
            </div>
            <div
              className="p-4 rounded-xl flex flex-col gap-2 bg-neutral-50 dark:bg-white/5 border border-neutral-200 dark:border-white/10"
            >
              <span className="text-xs uppercase tracking-wide text-neutral-500 dark:text-neutral-400">Notes rapides</span>
              <ul className="text-sm text-neutral-600 dark:text-neutral-300 space-y-1 list-disc list-inside">
                <li>ECE mesure l’écart entre proba prévues et résultats réels.</li>
                <li>Un Brier plus bas indique une calibration plus fiable.</li>
                <li>Relancer la calibration rafraîchit les courbes de fiabilité.</li>
              </ul>
            </div>
          </div>
        </div>
      </GlassCard>

      <div className="fixed top-4 right-4 z-50 space-y-3">
        <AnimatePresence>
          {toasts.map((toast) => {
            const tone = toastTone[toast.type] || toastTone.info;
            const Icon = tone.icon;
            return (
              <motion.div
                key={toast.id}
                initial={{ opacity: 0, x: 50, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 50, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className={`glass-notification ${tone.className}`}
              >
                <div className="flex items-start gap-3">
                  <Icon size={18} className={`${tone.color} flex-shrink-0 mt-0.5`} />
                  <div className="text-sm text-neutral-900 dark:text-neutral-100">{toast.message}</div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

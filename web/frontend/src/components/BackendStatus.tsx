import { memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, CheckCircle, Loader2, RefreshCw, WifiOff } from 'lucide-react';
import { useBackendHealth } from '../hooks/useBackendHealth';

interface BackendStatusProps {
  /** Afficher le composant même si le backend est healthy */
  showWhenHealthy?: boolean;
  /** Position du banner */
  position?: 'top' | 'bottom';
  /** Callback quand le statut change */
  onStatusChange?: (isHealthy: boolean) => void;
}

/**
 * Composant de smoke check pour afficher l'état du backend
 *
 * Affiche un banner "Backend OK" ou une alerte si le backend est indisponible.
 *
 * @example
 * ```tsx
 * // Dans App.jsx ou un layout
 * <BackendStatus showWhenHealthy={false} />
 * ```
 */
export const BackendStatus = memo(({
  showWhenHealthy = false,
  position = 'top',
  onStatusChange,
}: BackendStatusProps) => {
  const { health, isHealthy, isChecking, checkHealth } = useBackendHealth({
    pollInterval: 30000, // Vérifier toutes les 30s
    timeout: 5000,
  });

  // Callback de changement de statut
  if (onStatusChange && health.status !== 'unknown' && health.status !== 'checking') {
    onStatusChange(isHealthy);
  }

  // Ne rien afficher si healthy et showWhenHealthy=false
  if (isHealthy && !showWhenHealthy) {
    return null;
  }

  const positionClasses = position === 'top'
    ? 'top-0 left-0 right-0'
    : 'bottom-0 left-0 right-0';

  return (
    <AnimatePresence>
      {(isHealthy && showWhenHealthy) && (
        <motion.div
          initial={{ opacity: 0, y: position === 'top' ? -50 : 50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: position === 'top' ? -50 : 50 }}
          className={`fixed ${positionClasses} z-50 px-4 py-2`}
        >
          <div className="max-w-md mx-auto flex items-center gap-3 px-4 py-2 rounded-lg shadow-lg"
            style={{
              background: 'rgba(46, 213, 115, 0.15)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(46, 213, 115, 0.3)',
            }}
          >
            <CheckCircle size={18} className="text-green-500 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <span className="text-sm font-medium text-green-400">
                Backend OK
              </span>
              {health.latencyMs !== undefined && (
                <span className="text-xs text-green-500/70 ml-2">
                  ({health.latencyMs}ms)
                </span>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {isChecking && (
        <motion.div
          initial={{ opacity: 0, y: position === 'top' ? -50 : 50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: position === 'top' ? -50 : 50 }}
          className={`fixed ${positionClasses} z-50 px-4 py-2`}
        >
          <div className="max-w-md mx-auto flex items-center gap-3 px-4 py-2 rounded-lg shadow-lg"
            style={{
              background: 'rgba(59, 130, 246, 0.15)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
            }}
          >
            <Loader2 size={18} className="text-blue-500 animate-spin flex-shrink-0" />
            <span className="text-sm font-medium text-blue-400">
              Vérification du backend...
            </span>
          </div>
        </motion.div>
      )}

      {!isHealthy && !isChecking && health.status !== 'unknown' && (
        <motion.div
          initial={{ opacity: 0, y: position === 'top' ? -50 : 50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: position === 'top' ? -50 : 50 }}
          className={`fixed ${positionClasses} z-50 px-4 py-2`}
        >
          <div className="max-w-lg mx-auto flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg"
            style={{
              background: 'rgba(220, 38, 38, 0.15)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(220, 38, 38, 0.3)',
            }}
          >
            <WifiOff size={20} className="text-red-500 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <AlertCircle size={16} className="text-red-400" />
                <span className="text-sm font-semibold text-red-400">
                  Backend indisponible
                </span>
              </div>
              <p className="text-xs text-red-500/80 mt-0.5">
                {health.error || 'Impossible de contacter le serveur'}
                {health.latencyMs !== undefined && ` (${health.latencyMs}ms)`}
              </p>
            </div>
            <motion.button
              onClick={() => checkHealth()}
              className="p-2 rounded-lg hover:bg-red-500/20 transition-colors"
              title="Réessayer"
            >
              <RefreshCw size={16} className="text-red-400" />
            </motion.button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
});

BackendStatus.displayName = 'BackendStatus';

export default BackendStatus;

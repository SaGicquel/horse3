import { useState, useEffect, useCallback, useRef } from 'react';
import { List } from 'react-window';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { chevauxAPI, coursesAPI } from '../services/api';
import {
  Search, 
  Activity, 
  Trophy, 
  Calendar, 
  ChevronUp, 
  ChevronDown, 
  X, 
  RefreshCw, 
  Database,
  ArrowRight,
  Loader2,
  ShieldAlert
} from 'lucide-react';
import { GlassCard } from '../components/GlassCard';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.05 }
  }
};

const rowVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: { opacity: 1, x: 0 }
};

const modalVariants = {
  hidden: { opacity: 0, scale: 0.95, y: 20 },
  visible: { 
    opacity: 1, 
    scale: 1, 
    y: 0,
    transition: { type: "spring", damping: 25, stiffness: 300 }
  },
  exit: { opacity: 0, scale: 0.95, y: 20 }
};

const ROLE_STORAGE_KEY = 'horse3-role';
const DEFAULT_ROLE = (import.meta.env.VITE_DEFAULT_ROLE || 'viewer').toLowerCase();
const ADMIN_TOKEN = import.meta.env.VITE_ADMIN_TOKEN;
const ROW_HEIGHT = 68;
const VIRTUALIZATION_THRESHOLD = 10000;
const chevalGridTemplate = '1.6fr 0.7fr 1fr 0.9fr 1fr 0.9fr 1.2fr';
const courseGridTemplate = '1fr 1.1fr 1.2fr 0.9fr 0.9fr';

const computeListHeight = () => {
  if (typeof window === 'undefined') return 640;
  return Math.max(320, Math.min(900, window.innerHeight - 280));
};

const getInitialRole = (search = '') => {
  if (typeof window === 'undefined') return DEFAULT_ROLE;
  const params = new URLSearchParams(search || window.location.search);
  const roleFromUrl = params.get('role');
  if (roleFromUrl) {
    const normalized = roleFromUrl.toLowerCase();
    localStorage.setItem(ROLE_STORAGE_KEY, normalized);
    return normalized;
  }

  const stored = localStorage.getItem(ROLE_STORAGE_KEY);
  return (stored || DEFAULT_ROLE).toLowerCase();
};

const Data = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // State
  const [userRole, setUserRole] = useState(() => getInitialRole(location.search));
  const [accessCode, setAccessCode] = useState('');
  const [accessError, setAccessError] = useState('');
  const [activeTab, setActiveTab] = useState('chevaux');
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [pagination, setPagination] = useState({ offset: 0, limit: 50, total: 0, hasMore: true });
  const [selectedItem, setSelectedItem] = useState(null);
  const [listHeight, setListHeight] = useState(computeListHeight());
  const isAdmin = userRole === 'admin';
  const shouldVirtualize = pagination.total > VIRTUALIZATION_THRESHOLD;
  
  // Refs
  const observerTarget = useRef(null);
  const searchTimeout = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const roleFromUrl = params.get('role');
    if (roleFromUrl) {
      const normalized = roleFromUrl.toLowerCase();
      setUserRole(normalized);
      localStorage.setItem(ROLE_STORAGE_KEY, normalized);
    }
  }, [location.search]);

  useEffect(() => {
    const handleResize = () => setListHeight(computeListHeight());
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const loadData = useCallback(async (offset, isReset = false) => {
    if (!isAdmin) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      setLoading(true);
      setError(null);

      const apiCall = activeTab === 'chevaux' 
        ? chevauxAPI.getAllChevaux 
        : coursesAPI.getCoursesVues;
      
      const response = await apiCall(
        pagination.limit, 
        offset, 
        sortConfig.key, 
        sortConfig.direction, 
        searchTerm,
        controller.signal
      );

      const newItems = activeTab === 'chevaux' ? (response.chevaux || []) : (response.courses || []);
      const totalItems = response.total || 0;

      if (isReset) {
        setData(newItems);
      } else {
        setData(prev => [...prev, ...newItems]);
      }
      
      setPagination(prev => ({
        ...prev,
        offset: offset + prev.limit,
        total: totalItems,
        hasMore: (offset + prev.limit) < totalItems
      }));

    } catch (err) {
      if (err?.name === 'CanceledError' || err?.code === 'ERR_CANCELED') {
        return;
      }
      setError('Impossible de charger les données. Veuillez réessayer.');
    } finally {
      setLoading(false);
    }
  }, [activeTab, isAdmin, pagination.limit, searchTerm, sortConfig.direction, sortConfig.key]);

  const resetAndLoad = useCallback(() => {
    if (!isAdmin) return;
    setData([]);
    setPagination(prev => ({ ...prev, offset: 0, hasMore: true, total: 0 }));
    setError(null);
    loadData(0, true);
  }, [isAdmin, loadData]);

  // Initial Load & Tab Change
  useEffect(() => {
    if (!isAdmin) return;
    resetAndLoad();
  }, [activeTab, isAdmin]);

  // Search Debounce
  useEffect(() => {
    if (!isAdmin) return;
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    
    searchTimeout.current = setTimeout(() => {
      resetAndLoad();
    }, 500);

    return () => {
      if (searchTimeout.current) clearTimeout(searchTimeout.current);
    };
  }, [searchTerm, isAdmin, resetAndLoad]);

  const handleSort = (key) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  // Effect for Sort changes
  useEffect(() => {
    // Avoid double load on initial mount (handled by activeTab effect)
    if (!isAdmin) return;
    if (pagination.offset !== 0 || data.length > 0) {
      resetAndLoad();
    }
  }, [sortConfig, isAdmin, pagination.offset, data.length, resetAndLoad]);

  // Infinite Scroll
  const handleObserver = useCallback((entries) => {
    if (!isAdmin || shouldVirtualize) return;
    const [target] = entries;
    if (target.isIntersecting && pagination.hasMore && !loading) {
      loadData(pagination.offset);
    }
  }, [isAdmin, shouldVirtualize, pagination.hasMore, pagination.offset, loading, loadData]);

  useEffect(() => {
    if (!isAdmin || shouldVirtualize) return;
    const observer = new IntersectionObserver(handleObserver, { threshold: 0.1 });
    if (observerTarget.current) observer.observe(observerTarget.current);
    return () => observer.disconnect();
  }, [handleObserver, isAdmin, shouldVirtualize]);

  const handleVirtualItemsRendered = useCallback(({ visibleStopIndex }) => {
    if (!shouldVirtualize || !isAdmin) return;
    if (visibleStopIndex >= data.length - 5 && pagination.hasMore && !loading) {
      loadData(pagination.offset);
    }
  }, [shouldVirtualize, isAdmin, data.length, pagination.hasMore, pagination.offset, loading, loadData]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const handleAdminUnlock = (event) => {
    event?.preventDefault();
    const normalized = accessCode.trim();
    if (!ADMIN_TOKEN && !normalized) {
      setAccessError('Saisissez le code admin pour continuer.');
      return;
    }
    if (ADMIN_TOKEN && normalized !== ADMIN_TOKEN) {
      setAccessError('Code administrateur invalide.');
      return;
    }
    setAccessError('');
    setUserRole('admin');
    localStorage.setItem(ROLE_STORAGE_KEY, 'admin');
  };

  const handleEntityNavigate = (type, value, event) => {
    event?.stopPropagation();
    if (!value) return;
    const encoded = encodeURIComponent(value);
    const routes = {
      cheval: `/cheval/${encoded}`,
      jockey: `/jockey/${encoded}`,
      entraineur: `/entraineur/${encoded}`,
      hippodrome: `/hippodrome/${encoded}`
    };
    const target = routes[type];
    if (target) {
      navigate(target);
    }
  };

  // Render Helpers
  const renderSortIcon = (key) => {
    if (sortConfig.key !== key) return <div className="w-4 h-4" />;
    return sortConfig.direction === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />;
  };

  const formatValue = (value, type = 'text') => {
    if (value === null || value === undefined) return '-';
    if (type === 'percent') {
      const num = parseFloat(value);
      return isNaN(num) ? '-' : `${num.toFixed(1)}%`;
    }
    if (type === 'date') {
      try {
        return new Date(value).toLocaleDateString('fr-FR');
      } catch {
        return value;
      }
    }
    return value;
  };

  const renderHorseCells = (item, Wrapper = 'div') => (
    <>
      <Wrapper className="p-4">
        <button
          type="button"
          onClick={(e) => handleEntityNavigate('cheval', item.nom, e)}
          className="flex items-center gap-2 text-[var(--color-primary)] font-semibold hover:underline"
        >
          <span className="truncate">{item.nom}</span>
          <ArrowRight size={14} className="opacity-70" />
        </button>
      </Wrapper>
      <Wrapper className="p-4 text-[var(--color-muted)]">{item.sexe || '-'}</Wrapper>
      <Wrapper className="p-4 text-[var(--color-muted)]">{item.race || '-'}</Wrapper>
      <Wrapper className="p-4 text-right font-mono">{item.nombre_courses_total || 0}</Wrapper>
      <Wrapper className="p-4 text-right font-mono text-green-600 dark:text-green-400">{item.nombre_victoires_total || 0}</Wrapper>
      <Wrapper className="p-4 text-right font-mono">
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          (item.taux_victoire || 0) > 20 ? 'bg-green-500/10 text-green-500' :
          (item.taux_victoire || 0) > 10 ? 'bg-yellow-500/10 text-yellow-500' :
          'bg-gray-500/10 text-gray-500'
        }`}>
          {formatValue(item.taux_victoire || 0, 'percent')}
        </span>
      </Wrapper>
      <Wrapper className="p-4 text-xs font-mono text-[var(--color-muted)] truncate">
        {item.dernier_resultat || '-'}
      </Wrapper>
    </>
  );

  const renderCourseCells = (item, Wrapper = 'div') => (
    <>
      <Wrapper className="p-4 whitespace-nowrap">{formatValue(item.date_course, 'date')}</Wrapper>
      <Wrapper className="p-4">
        <button
          type="button"
          onClick={(e) => handleEntityNavigate('hippodrome', item.hippodrome, e)}
          className="flex items-center gap-2 text-left text-[var(--color-text)] hover:text-[var(--color-primary)] hover:underline"
        >
          <span className="truncate">{item.hippodrome || '-'}</span>
          <ArrowRight size={14} className="opacity-60" />
        </button>
      </Wrapper>
      <Wrapper className="p-4 font-medium text-[var(--color-primary)]">
        <button
          type="button"
          onClick={(e) => handleEntityNavigate('cheval', item.nom_cheval, e)}
          className="flex items-center gap-2 hover:underline"
        >
          <span className="truncate">{item.nom_cheval || '-'}</span>
          <ArrowRight size={14} className="opacity-60" />
        </button>
      </Wrapper>
      <Wrapper className="p-4 text-center text-[var(--color-muted)]">{item.reunion} {item.course}</Wrapper>
      <Wrapper className="p-4 text-center">
        {item.victoire ? (
          <span 
            className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-500/10 text-green-500"
          >
            <Trophy size={12} /> Victoire
          </span>
        ) : (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-500/10 text-gray-400">
            -
          </span>
        )}
      </Wrapper>
    </>
  );

  const gridTemplate = activeTab === 'chevaux' ? chevalGridTemplate : courseGridTemplate;

  const columns = activeTab === 'chevaux' ? [
    { key: 'nom', label: 'Nom', sortable: true },
    { key: 'sexe', label: 'Sexe', sortable: true },
    { key: 'race', label: 'Race', sortable: true },
    { key: 'nb_courses', label: 'Courses', sortable: true, align: 'right' },
    { key: 'nb_victoires', label: 'Victoires', sortable: true, align: 'right' },
    { key: 'taux_victoire', label: 'Taux', sortable: true, align: 'right' },
    { key: 'dernier_resultat', label: 'Musique', sortable: false }
  ] : [
    { key: 'date_course', label: 'Date', sortable: true },
    { key: 'hippodrome', label: 'Hippodrome', sortable: true },
    { key: 'nom_cheval', label: 'Cheval', sortable: true },
    { key: 'reunion', label: 'R/C', sortable: true, align: 'center' },
    { key: 'victoire', label: 'Résultat', sortable: true, align: 'center' }
  ];

  const VirtualizedRow = ({ index, style, data: rowData }) => {
    const item = rowData.items[index];
    if (!item) return null;
    const rowStyle = { ...style, width: '100%', display: 'grid', gridTemplateColumns: rowData.gridTemplate };
    const rowClass = `border-b border-[var(--color-border)] ${rowData.activeTab === 'chevaux' ? 'cursor-pointer' : ''} hover:bg-[var(--color-secondary)]/50 transition-colors`;
    return (
      <div
        style={rowStyle}
        className={rowClass}
        onClick={() => rowData.activeTab === 'chevaux' ? rowData.onSelect(item) : undefined}
      >
        {rowData.activeTab === 'chevaux' ? renderHorseCells(item, 'div') : renderCourseCells(item, 'div')}
      </div>
    );
  };

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-[var(--color-bg)] p-4 sm:p-8 font-sans text-[var(--color-text)]">
        <div className="max-w-3xl mx-auto">
          <GlassCard className="p-8">
            <div className="flex flex-col gap-4">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-full bg-red-500/10 text-red-400">
                  <ShieldAlert size={24} />
                </div>
                <div className="flex-1 space-y-2">
                  <h1 className="text-2xl font-bold">Accès restreint</h1>
                  <p className="text-[var(--color-muted)]">
                    Cette section est réservée aux administrateurs. Authentifiez-vous pour explorer l&apos;entrepôt de données.
                  </p>
                </div>
              </div>

              <form className="flex flex-col sm:flex-row gap-3" onSubmit={handleAdminUnlock}>
                <input
                  type="password"
                  value={accessCode}
                  onChange={(e) => setAccessCode(e.target.value)}
                  placeholder="Code admin ou token"
                  className="flex-1 px-4 py-3 rounded-xl bg-[var(--color-bg)] border border-[var(--color-border)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
                <button
                  type="submit"
                  className="px-5 py-3 bg-[var(--color-primary)] text-white rounded-xl font-medium hover:opacity-90 transition"
                >
                  Déverrouiller
                </button>
              </form>

              {accessError && <p className="text-sm text-red-400">{accessError}</p>}
              <p className="text-xs text-[var(--color-muted)]">
                Rôle actuel : <span className="text-[var(--color-primary)] font-semibold">{userRole}</span>. Ajoutez <code>?role=admin</code> à l&apos;URL ou stockez <code>horse3-role=admin</code> dans votre navigateur si votre session est déjà validée.
              </p>
            </div>
          </GlassCard>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg)] p-4 sm:p-8 font-sans text-[var(--color-text)]">
      
      {/* Header Section */}
      <div className="max-w-7xl mx-auto mb-8">
        <motion.div 
          className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
              >
                <Database className="text-[var(--color-primary)]" />
              </motion.div>
              Explorateur de Données
            </h1>
            <p className="text-[var(--color-muted)] mt-1">
              Consultez et analysez l'ensemble de votre base de données hippiques
            </p>
            <div className="inline-flex items-center gap-2 mt-2 px-3 py-1 rounded-full bg-[var(--color-secondary)]/60 border border-[var(--color-border)] text-xs text-[var(--color-primary)]">
              <ShieldAlert size={14} />
              Accès admin
            </div>
          </div>
          
          <motion.div 
            className="flex items-center gap-2 p-1 rounded-xl"
            style={{
              background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.55)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid rgba(var(--color-border-rgb), 0.1)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.6)'
            }}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <motion.button
              onClick={() => setActiveTab('chevaux')}
              className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'chevaux' 
                  ? 'bg-[var(--color-primary)] text-white shadow-lg' 
                  : 'text-[var(--color-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-secondary)]'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Activity size={16} />
              Chevaux
            </motion.button>
            <motion.button
              onClick={() => setActiveTab('courses')}
              className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'courses' 
                  ? 'bg-[var(--color-primary)] text-white shadow-lg' 
                  : 'text-[var(--color-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-secondary)]'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Calendar size={16} />
              Courses
            </motion.button>
          </motion.div>
        </motion.div>

        {/* Controls Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <GlassCard className="p-4 mb-6">
            <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-muted)]" size={18} />
              <input
                type="text"
                placeholder={`Rechercher dans ${activeTab}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-lg bg-[var(--color-bg)] border border-[var(--color-border)] focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent outline-none transition-all"
              />
              {searchTerm && (
                <button 
                  onClick={() => setSearchTerm('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--color-muted)] hover:text-[var(--color-text)]"
                >
                  <X size={16} />
                </button>
              )}
            </div>
            
            <div className="flex items-center gap-3">
              <motion.div 
                className="px-4 py-2.5 rounded-lg bg-[var(--color-bg)] border border-[var(--color-border)] text-sm text-[var(--color-muted)] whitespace-nowrap"
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                key={pagination.total}
              >
                <span className="font-semibold text-[var(--color-text)]">{pagination.total}</span> résultats
              </motion.div>
              {shouldVirtualize && (
                <div className="px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/30 text-xs font-semibold text-green-500">
                  Virtualisation 10k+
                </div>
              )}
              <motion.button 
                onClick={resetAndLoad}
                className="p-2.5 rounded-lg bg-[var(--color-bg)] border border-[var(--color-border)] text-[var(--color-muted)] hover:text-[var(--color-primary)] hover:border-[var(--color-primary)] transition-all"
                title="Actualiser"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
              </motion.button>
            </div>
          </div>
          </GlassCard>
        </motion.div>

        {/* Data Table */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <GlassCard className="overflow-hidden">
            {error ? (
              <motion.div 
                className="p-12 text-center"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <motion.div 
                  className="text-red-500 mb-4 flex justify-center"
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 0.5 }}
                >
                  <X size={48} />
                </motion.div>
                <h3 className="text-lg font-semibold mb-2">Une erreur est survenue</h3>
                <p className="text-[var(--color-muted)] mb-6">{error}</p>
                <motion.button 
                  onClick={resetAndLoad}
                  className="px-6 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:opacity-90 transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Réessayer
                </motion.button>
              </motion.div>
            ) : (
              <div className="overflow-x-auto">
                <div className="min-w-[960px]">
                  <div 
                    className="grid text-sm font-semibold border-b"
                    style={{
                      gridTemplateColumns: gridTemplate,
                      background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.4)',
                      backdropFilter: 'blur(16px)',
                      borderColor: 'rgba(var(--color-border-rgb), 0.1)'
                    }}
                  >
                    {columns.map((column) => (
                      <button
                        key={column.key}
                        type="button"
                        onClick={() => column.sortable && handleSort(column.key)}
                        className={`p-4 text-left font-semibold flex items-center gap-2 transition-colors ${column.align === 'right' ? 'justify-end' : column.align === 'center' ? 'justify-center' : ''} ${column.sortable ? 'hover:bg-[var(--color-border)] cursor-pointer' : ''}`}
                      >
                        <span>{column.label}</span>
                        {column.sortable && renderSortIcon(column.key)}
                      </button>
                    ))}
                  </div>

                  {data.length === 0 && !loading ? (
                    <div className="p-12 text-center text-[var(--color-muted)]">
                      <div className="text-4xl mb-2">📭</div>
                      Aucune donnée trouvée
                    </div>
                  ) : shouldVirtualize ? (
                    <List
                      height={listHeight}
                      itemCount={data.length}
                      itemSize={ROW_HEIGHT}
                      width="100%"
                      itemKey={(index, rowData) => rowData.items[index]?.race_key || rowData.items[index]?.nom || index}
                      onItemsRendered={handleVirtualItemsRendered}
                      itemData={{ items: data, activeTab, onSelect: setSelectedItem, gridTemplate }}
                    >
                      {VirtualizedRow}
                    </List>
                  ) : (
                    <div>
                      {data.map((item, idx) => (
                        <div 
                          key={item.race_key || item.nom || item.id || idx} 
                          onClick={() => activeTab === 'chevaux' ? setSelectedItem(item) : undefined}
                          className={`grid items-center border-b border-[var(--color-border)] hover:bg-[var(--color-secondary)]/50 transition-colors ${activeTab === 'chevaux' ? 'cursor-pointer' : ''}`}
                          style={{ gridTemplateColumns: gridTemplate }}
                        >
                          {activeTab === 'chevaux' ? renderHorseCells(item) : renderCourseCells(item)}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* Loading Indicator / Infinite Scroll Target */}
            {!shouldVirtualize && (
              <div ref={observerTarget} className="p-6 flex justify-center">
                {loading && (
                  <motion.div 
                    className="flex items-center gap-3 text-[var(--color-muted)]"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <Loader2 className="w-5 h-5 animate-spin text-[var(--color-primary)]" />
                    <span>Chargement...</span>
                  </motion.div>
                )}
              </div>
            )}
            {shouldVirtualize && loading && (
              <div className="p-4 flex justify-center">
                <motion.div 
                  className="flex items-center gap-3 text-[var(--color-muted)]"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <Loader2 className="w-5 h-5 animate-spin text-[var(--color-primary)]" />
                  <span>Chargement...</span>
                </motion.div>
              </div>
            )}
          </GlassCard>
        </motion.div>
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedItem && (
          <motion.div 
            className="fixed inset-0 bg-black/40 backdrop-blur-md z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedItem(null)}
          >
            <motion.div 
              className="rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto"
              style={{
                background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.85)',
                backdropFilter: 'blur(32px)',
                WebkitBackdropFilter: 'blur(32px)',
                border: '1px solid rgba(var(--color-border-rgb), 0.15)',
                boxShadow: '0 24px 80px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.8)'
              }}
              variants={modalVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-6 border-b flex justify-between items-start" style={{ borderColor: 'rgba(var(--color-border-rgb), 0.1)' }}>
                <div>
                  <h2 className="text-2xl font-bold text-[var(--color-text)] flex items-center gap-3">
                    <motion.div
                      animate={{ rotate: [0, 10, -10, 0] }}
                      transition={{ duration: 0.5 }}
                    >
                      <Activity className="text-[var(--color-primary)]" />
                    </motion.div>
                    {selectedItem.nom}
                  </h2>
                  <p className="text-[var(--color-muted)] mt-1">{selectedItem.race || '-'} • {selectedItem.sexe || '-'}</p>
                </div>
                <motion.button 
                  onClick={() => setSelectedItem(null)} 
                  className="p-2 hover:bg-[var(--color-secondary)] rounded-full transition-colors"
                  whileHover={{ scale: 1.1, rotate: 90 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <X size={20} />
                </motion.button>
              </div>
              
              <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                <motion.div 
                  className="p-5 rounded-xl text-center"
                  style={{ 
                    background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.5)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(var(--color-border-rgb), 0.1)',
                    boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.5)'
                  }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                >
                  <div className="text-[var(--color-muted)] text-xs uppercase tracking-wider mb-2">Courses</div>
                  <div className="text-3xl font-bold">{selectedItem.nombre_courses_total || 0}</div>
                </motion.div>
                <motion.div 
                  className="p-5 rounded-xl text-center"
                  style={{ 
                    background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.5)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(var(--color-border-rgb), 0.1)',
                    boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.5)'
                  }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <div className="text-[var(--color-muted)] text-xs uppercase tracking-wider mb-2">Victoires</div>
                  <div className="text-3xl font-bold text-green-500">{selectedItem.nombre_victoires_total || 0}</div>
                </motion.div>
                <motion.div 
                  className="p-5 rounded-xl text-center"
                  style={{ 
                    background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.5)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(var(--color-border-rgb), 0.1)',
                    boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.5)'
                  }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="text-[var(--color-muted)] text-xs uppercase tracking-wider mb-2">Réussite</div>
                  <div className="text-3xl font-bold text-[var(--color-primary)]">{formatValue(selectedItem.taux_victoire || 0, 'percent')}</div>
                </motion.div>
                
                <motion.div 
                  className="col-span-full mt-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Activity size={16} />
                    Musique
                  </h3>
                  <div className="bg-[var(--color-bg)] p-4 rounded-xl border border-[var(--color-border)] font-mono text-sm break-all">
                    {selectedItem.dernier_resultat || "Aucune musique disponible"}
                  </div>
                </motion.div>
              </div>
              
              <div className="p-6 border-t border-[var(--color-border)] bg-[var(--color-secondary)]/30 flex justify-end">
                <motion.button 
                  onClick={() => setSelectedItem(null)}
                  className="px-6 py-2.5 bg-[var(--color-card)] border border-[var(--color-border)] rounded-xl hover:bg-[var(--color-secondary)] transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Fermer
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Data;

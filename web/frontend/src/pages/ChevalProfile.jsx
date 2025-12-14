import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  TrophyIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  MagnifyingGlassIcon,
  FireIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useNavigate, useParams } from 'react-router-dom';
import { GlassCard } from '../components/GlassCard';
import EntityHeader from '../components/EntityHeader';
import { API_BASE } from '../config/api';

export default function ChevalProfile() {
  const [searchTerm, setSearchTerm] = useState('');
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [topChevaux, setTopChevaux] = useState([]);
  const [loadingTop, setLoadingTop] = useState(true);
  const navigate = useNavigate();
  const { id } = useParams();
  const formatCurrency = (value = 0) => `${Math.round((value || 0) / 100).toLocaleString('fr-FR')}€`;
  const formatPercent = (value) => value === undefined || value === null ? '--' : `${Number(value).toFixed(1)}%`;

  // Charger les top chevaux au montage
  useEffect(() => {
    loadTopChevaux();
  }, []);

  const loadTopChevaux = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/chevaux?limit=12&sort_by=nombre_victoires_total&sort_order=desc`);
      if (response.ok) {
        const data = await response.json();
        setTopChevaux(data.chevaux || []);
      }
    } catch (err) {
      console.error('Erreur chargement top chevaux:', err);
    } finally {
      setLoadingTop(false);
    }
  };

  const searchCheval = async (nomCheval = null) => {
    const nom = nomCheval || searchTerm;
    if (!nom.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/cheval/${encodeURIComponent(nom)}/profile`);
      if (!response.ok) {
        throw new Error('Cheval non trouvé');
      }
      const data = await response.json();
      setProfile(data);
      if (nomCheval) setSearchTerm(nomCheval);
    } catch (err) {
      setError(err.message);
      setProfile(null);
    } finally {
      setLoading(false);
    }
  };

  // Charger automatiquement un cheval si l'URL contient un paramètre
  useEffect(() => {
    if (id) {
      const decoded = decodeURIComponent(id);
      setSearchTerm(decoded);
      searchCheval(decoded);
    }
  }, [id]);

  // Préparer les données pour les graphiques
  const historyChartData = profile?.historique?.map(h => ({
    date: h.date?.slice(5), // MM-DD
    place: h.place || 15,
    cote: h.cote,
    gains: h.gains || 0
  })).reverse() || [];

  const typePerformanceData = profile?.performance_par_type?.map(t => ({
    name: t.type,
    taux: t.taux,
    courses: t.courses,
    victoires: t.victoires
  })) || [];

  const recentMetrics = profile?.recent_metrics || {};

  const trendData = useMemo(() => {
    if (profile?.recent_performance?.length) {
      return profile.recent_performance.map((entry) => ({
        label: entry.date?.slice(5),
        value: Number(entry.taux ?? 0)
      }));
    }
    if (profile?.historique?.length) {
      const entries = [...profile.historique].slice(0, 12).reverse();
      return entries.map((h) => ({
        label: h.date?.slice(5),
        value: h.victoire ? 100 : h.place ? Math.max(0, 100 - h.place * 8) : 10
      }));
    }
    return [];
  }, [profile]);

  const headerMetrics = profile ? [
    { label: 'Taux réussite', value: formatPercent(profile.statistiques.taux_victoire) },
    { label: 'Gains', value: formatCurrency(profile.statistiques.gains_total) },
    { label: 'Forme 30j', value: formatPercent(recentMetrics.taux_victoire_30j ?? profile.statistiques.taux_victoire) },
    { label: 'Courses 30j', value: (recentMetrics.courses_30j ?? profile.statistiques.nb_courses)?.toString() || '0' }
  ] : [];

  const handleSeeCourses = () => {
    const chevalName = profile?.nom || searchTerm;
    navigate(chevalName ? `/courses?cheval=${encodeURIComponent(chevalName)}` : '/courses');
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
      {/* Header avec recherche */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 dark:from-purple-400 dark:to-pink-400 bg-clip-text text-transparent mb-2">
          🐴 Profil Cheval
        </h1>
        <p className="text-neutral-600 dark:text-neutral-400 mb-6">
          Analysez les performances détaillées d'un cheval
        </p>

        <div className="flex flex-col sm:flex-row justify-center gap-3">
          <div className="relative w-full sm:w-96">
            <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-neutral-400" />
            <input
              type="text"
              placeholder="Nom du cheval..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && searchCheval()}
              className="w-full pl-12 pr-4 py-3 bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-xl text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 dark:placeholder-neutral-400 focus:outline-none focus:border-purple-500/50 transition-colors"
            />
          </div>
          <button
            onClick={() => searchCheval()}
            disabled={loading}
            className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {loading ? 'Recherche...' : 'Rechercher'}
          </button>
        </div>
      </motion.div>

      {/* Message d'erreur */}
      {error && (
        <GlassCard className="text-center py-8 border border-red-500/30">
          <div className="text-4xl mb-2">❌</div>
          <p className="text-red-600 dark:text-red-400">{error}</p>
        </GlassCard>
      )}

      {/* Profil du cheval */}
      {profile && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6"
        >
          <EntityHeader
            name={profile.nom || searchTerm}
            emoji="🐴"
            subtitle="Synthèse et dernières performances"
            pill="Cheval"
            metrics={headerMetrics}
            trendData={trendData}
            onCta={handleSeeCourses}
            ctaLabel="Voir courses liées"
          />

          {/* Stats principales */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
            <GlassCard className="text-center p-4">
              <TrophyIcon className="h-8 w-8 mx-auto text-yellow-500 dark:text-yellow-400 mb-2" />
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">{profile.statistiques.victoires}</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Victoires</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <ChartBarIcon className="h-8 w-8 mx-auto text-blue-500 dark:text-blue-400 mb-2" />
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">{profile.statistiques.nb_courses}</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Courses</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <ArrowTrendingUpIcon className="h-8 w-8 mx-auto text-green-500 dark:text-green-400 mb-2" />
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">{profile.statistiques.taux_victoire}%</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Taux Victoire</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <div className="text-3xl mb-2">🥈</div>
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{profile.statistiques.taux_place}%</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Taux Placé</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <div className="text-3xl mb-2">📊</div>
              <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{profile.statistiques.cote_moyenne}</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Cote Moy.</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <div className="text-3xl mb-2">⭐</div>
              <div className="text-2xl font-bold text-pink-600 dark:text-pink-400">{profile.statistiques.meilleure_cote}</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Meilleure Cote</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <CurrencyDollarIcon className="h-8 w-8 mx-auto text-emerald-500 dark:text-emerald-400 mb-2" />
              <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{(profile.statistiques.gains_total / 100).toFixed(0)}€</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Gains Total</div>
            </GlassCard>

            <GlassCard className="text-center p-4">
              <div className="text-3xl mb-2">🏆</div>
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{(profile.statistiques.meilleur_gain / 100).toFixed(0)}€</div>
              <div className="text-xs text-neutral-600 dark:text-neutral-400">Meilleur Gain</div>
            </GlassCard>
          </div>

          {/* Graphiques */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Évolution des places */}
            <GlassCard className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">📈 Évolution des Places</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={historyChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="date" stroke="var(--color-text)" fontSize={12} />
                  <YAxis reversed domain={[1, 'auto']} stroke="var(--color-text)" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--color-card)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '8px',
                      color: 'var(--color-text)'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="place"
                    stroke="#8B5CF6"
                    strokeWidth={2}
                    dot={{ fill: '#8B5CF6', r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </GlassCard>

            {/* Performance par type */}
            <GlassCard className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">🏁 Performance par Type de Course</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={typePerformanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="name" stroke="var(--color-text)" fontSize={10} />
                  <YAxis stroke="var(--color-text)" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--color-card)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '8px',
                      color: 'var(--color-text)'
                    }}
                  />
                  <Bar dataKey="taux" fill="#10B981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </GlassCard>
          </div>

          {/* Historique des courses */}
          <GlassCard className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">📋 Historique des Courses</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-neutral-500 dark:text-neutral-400 text-sm border-b border-neutral-200 dark:border-white/10">
                    <th className="pb-3">Date</th>
                    <th className="pb-3">Hippodrome</th>
                    <th className="pb-3">Type</th>
                    <th className="pb-3">Distance</th>
                    <th className="pb-3">Cote</th>
                    <th className="pb-3">Place</th>
                    <th className="pb-3">Gains</th>
                    <th className="pb-3">Jockey</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {profile.historique.map((h, i) => (
                    <tr key={i} className="border-b border-neutral-200 dark:border-white/5 hover:bg-neutral-50 dark:hover:bg-white/5">
                      <td className="py-3 text-neutral-600 dark:text-neutral-300">{h.date}</td>
                      <td className="py-3 text-neutral-900 dark:text-neutral-100">{h.hippodrome}</td>
                      <td className="py-3 text-neutral-600 dark:text-neutral-400">{h.type_course}</td>
                      <td className="py-3 text-neutral-600 dark:text-neutral-400">{h.distance}m</td>
                      <td className="py-3 text-yellow-600 dark:text-yellow-400">{h.cote}</td>
                      <td className={`py-3 font-bold ${h.victoire ? 'text-green-600 dark:text-green-400' :
                          h.place && h.place <= 3 ? 'text-yellow-600 dark:text-yellow-400' : 'text-neutral-500 dark:text-neutral-400'
                        }`}>
                        {h.victoire ? '🏆 1er' : h.place ? `${h.place}ème` : '-'}
                      </td>
                      <td className="py-3 text-emerald-600 dark:text-emerald-400">
                        {h.gains ? `${(h.gains / 100).toFixed(0)}€` : '-'}
                      </td>
                      <td className="py-3 text-neutral-600 dark:text-neutral-400">{h.jockey}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* État initial - Top Chevaux */}
      {!profile && !error && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6"
        >
          {/* Top Chevaux */}
          <GlassCard className="p-6">
            <div className="flex items-center gap-2 mb-4">
              <FireIcon className="h-6 w-6 text-orange-500 dark:text-orange-400" />
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">🏆 Top Chevaux</h3>
              <span className="text-sm text-neutral-500 dark:text-neutral-400">(par victoires)</span>
            </div>

            {loadingTop ? (
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {[...Array(12)].map((_, i) => (
                  <div key={i} className="h-24 bg-neutral-100 dark:bg-white/5 rounded-xl animate-pulse" />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {topChevaux.map((cheval, index) => (
                  <motion.div
                    key={cheval.nom}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-white dark:bg-white/5 rounded-xl p-4 cursor-pointer hover:bg-neutral-50 dark:hover:bg-white/10 hover:scale-[1.02] transition-all border border-neutral-200 dark:border-transparent hover:border-purple-500/30 shadow-sm dark:shadow-none"
                    onClick={() => searchCheval(cheval.nom)}
                  >
                    <div className="text-center">
                      <div className="text-2xl mb-2">
                        {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🐴'}
                      </div>
                      <h4 className="text-neutral-900 dark:text-neutral-100 font-medium text-xs truncate">{cheval.nom}</h4>
                      <div className="flex justify-center gap-2 mt-2 text-xs">
                        <span className="text-green-600 dark:text-green-400">{cheval.nombre_victoires_total || 0} V</span>
                        <span className="text-neutral-500 dark:text-neutral-400">{cheval.nombre_courses_total || 0} C</span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </GlassCard>

          {/* Message explicatif */}
          <GlassCard className="text-center py-8 p-6">
            <div className="text-6xl mb-4">🔍</div>
            <p className="text-neutral-600 dark:text-neutral-400 text-lg">Recherchez un cheval ou cliquez sur un des top chevaux</p>
            <p className="text-neutral-500 dark:text-neutral-500 text-sm mt-2">Statistiques, historique, performances par type de course...</p>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}

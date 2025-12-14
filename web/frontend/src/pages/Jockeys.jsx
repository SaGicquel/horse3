import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  UserIcon,
  TrophyIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  MagnifyingGlassIcon,
  FireIcon
} from '@heroicons/react/24/outline';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import GlassCard from '../components/GlassCard';
import EntityHeader from '../components/EntityHeader';
import { useNavigate, useParams } from 'react-router-dom';
import { API_BASE } from '../config/api';

const COLORS = ['#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6'];

export default function Jockeys() {
  const [jockeys, setJockeys] = useState([]);
  const [topJockeys, setTopJockeys] = useState([]);
  const [selectedJockey, setSelectedJockey] = useState(null);
  const [jockeyStats, setJockeyStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();
  const { id } = useParams();
  const formatCurrency = (value = 0) => `${Math.round((value || 0) / 100).toLocaleString('fr-FR')}€`;
  const formatPercent = (value) => value === undefined || value === null ? '--' : `${Number(value).toFixed(1)}%`;

  useEffect(() => {
    loadTopJockeys();
  }, []);

  const loadTopJockeys = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/jockeys/top`);
      if (response.ok) {
        const data = await response.json();
        setTopJockeys(data.jockeys || []);
        setJockeys(data.jockeys || []);
      }
    } catch (error) {
      console.error('Erreur chargement jockeys:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadJockeyStats = async (nom) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/jockey/${encodeURIComponent(nom)}/stats`);
      if (response.ok) {
        const data = await response.json();
        setJockeyStats(data);
        setSelectedJockey(nom);
      }
    } catch (error) {
      console.error('Erreur chargement stats jockey:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) {
      const decoded = decodeURIComponent(id);
      setSearchTerm(decoded);
      loadJockeyStats(decoded);
    }
  }, [id]);

  const filteredJockeys = jockeys.filter(j =>
    j.nom?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Données graphiques
  const tendanceData = jockeyStats?.tendance_mensuelle?.map(t => ({
    mois: t.mois,
    victoires: t.victoires,
    courses: t.courses,
    taux: t.taux
  })) || [];

  const typeData = jockeyStats?.performance_par_type?.map(p => ({
    name: p.type,
    value: p.victoires,
    taux: p.taux
  })) || [];

  const hippodromeData = jockeyStats?.top_hippodromes?.map(h => ({
    name: h.nom,
    victoires: h.victoires,
    taux: h.taux
  })) || [];

  const recentMetrics = jockeyStats?.recent_metrics || {};

  const trendData = useMemo(() => {
    if (jockeyStats?.recent_performance?.length) {
      return jockeyStats.recent_performance.map((entry) => ({
        label: entry.date?.slice(5),
        value: Number(entry.taux ?? 0)
      }));
    }
    if (tendanceData.length) {
      return [...tendanceData].reverse().map((item) => ({
        label: item.mois?.slice(5) || item.mois,
        value: item.taux ?? 0
      }));
    }
    return [];
  }, [jockeyStats, tendanceData]);

  const headerMetrics = jockeyStats ? [
    { label: 'Taux réussite', value: formatPercent(jockeyStats.statistiques.taux_victoire) },
    { label: 'Gains', value: formatCurrency(jockeyStats.statistiques.gains_total || 0) },
    { label: 'Forme 30j', value: formatPercent(recentMetrics.taux_victoire_30j ?? jockeyStats.statistiques.taux_victoire) },
    { label: 'Courses 30j', value: (recentMetrics.courses_30j ?? jockeyStats.statistiques.courses)?.toString() || '0' }
  ] : [];

  const handleSeeCourses = () => {
    const jockeyName = jockeyStats?.nom || selectedJockey || searchTerm;
    navigate(jockeyName ? `/courses?jockey=${encodeURIComponent(jockeyName)}` : '/courses');
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
          👨‍✈️ Jockeys
        </h1>
        <p className="text-gray-400">
          Statistiques et performances des jockeys
        </p>
      </motion.div>

      {/* Recherche */}
      <div className="flex justify-center">
        <div className="relative w-full max-w-md">
          <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Rechercher un jockey..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-white/5 dark:bg-white/5 border border-white/10 dark:border-white/10 rounded-xl text-[var(--color-text)] placeholder-gray-500 focus:outline-none focus:border-blue-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Top Jockeys en forme */}
      <GlassCard>
        <div className="flex items-center gap-2 mb-4">
          <FireIcon className="h-6 w-6 text-orange-400" />
          <h3 className="text-lg font-semibold text-[var(--color-text)]">Top Jockeys</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {(searchTerm ? filteredJockeys : topJockeys).slice(0, 12).map((jockey, index) => (
            <motion.div
              key={jockey.nom}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className={`bg-white/5 rounded-xl p-4 cursor-pointer hover:bg-white/10 transition-all ${
                selectedJockey === jockey.nom ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => loadJockeyStats(jockey.nom)}
            >
              <div className="text-center">
                <div className={`text-3xl mb-2 ${
                  index === 0 ? '' : index === 1 ? '' : index === 2 ? '' : ''
                }`}>
                  {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '👨‍✈️'}
                </div>
                <h4 className="text-[var(--color-text)] font-medium text-sm truncate">{jockey.nom}</h4>
                <div className="flex justify-center gap-2 mt-2 text-xs">
                  <span className="text-green-400">{jockey.victoires} V</span>
                  <span className="text-gray-400">{jockey.taux}%</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </GlassCard>

      {/* Stats du jockey sélectionné */}
      {jockeyStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <EntityHeader
            name={jockeyStats.nom}
            emoji="👨‍✈️"
            subtitle="Synthèse des 30 derniers jours"
            pill="Jockey"
            metrics={headerMetrics}
            trendData={trendData}
            onCta={handleSeeCourses}
            ctaLabel="Voir courses liées"
          />

          {/* Stats principales */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <GlassCard className="text-center">
              <TrophyIcon className="h-8 w-8 mx-auto text-yellow-400 mb-2" />
              <div className="text-2xl font-bold text-[var(--color-text)]">{jockeyStats.statistiques.victoires}</div>
              <div className="text-xs text-gray-400">Victoires</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <ChartBarIcon className="h-8 w-8 mx-auto text-blue-400 mb-2" />
              <div className="text-2xl font-bold text-[var(--color-text)]">{jockeyStats.statistiques.courses}</div>
              <div className="text-xs text-gray-400">Courses</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <ArrowTrendingUpIcon className="h-8 w-8 mx-auto text-green-400 mb-2" />
              <div className="text-2xl font-bold text-green-400">{jockeyStats.statistiques.taux_victoire}%</div>
              <div className="text-xs text-gray-400">Taux Victoire</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <div className="text-3xl mb-2">🥈</div>
              <div className="text-2xl font-bold text-purple-400">{jockeyStats.statistiques.taux_place}%</div>
              <div className="text-xs text-gray-400">Taux Placé</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <div className="text-3xl mb-2">💰</div>
              <div className="text-2xl font-bold text-emerald-400">{jockeyStats.statistiques.cote_moyenne}</div>
              <div className="text-xs text-gray-400">Cote Moy. Victoire</div>
            </GlassCard>
          </div>

          {/* Graphiques */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Tendance mensuelle */}
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">📈 Tendance Mensuelle</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={tendanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="mois" stroke="#9CA3AF" fontSize={10} />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="taux" 
                    stroke="#8B5CF6" 
                    strokeWidth={2}
                    dot={{ fill: '#8B5CF6', r: 4 }}
                    name="Taux %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </GlassCard>

            {/* Performance par type */}
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🏁 Par Type de Course</h3>
              {typeData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={typeData}
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, taux }) => `${name}: ${taux}%`}
                    >
                      {typeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex flex-col items-center justify-center h-[250px] text-gray-500">
                  <div className="text-4xl mb-2">📊</div>
                  <p>Données non disponibles</p>
                </div>
              )}
            </GlassCard>

            {/* Top hippodromes */}
            <GlassCard className="lg:col-span-2">
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🏟️ Meilleurs Hippodromes</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={hippodromeData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9CA3AF" fontSize={12} />
                  <YAxis dataKey="name" type="category" stroke="#9CA3AF" fontSize={10} width={100} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="victoires" fill="#10B981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </GlassCard>
          </div>

          {/* Associations gagnantes */}
          {jockeyStats.associations_gagnantes && jockeyStats.associations_gagnantes.length > 0 && (
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🤝 Meilleures Associations Jockey/Entraineur</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {jockeyStats.associations_gagnantes.slice(0, 6).map((assoc, i) => (
                  <div key={i} className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className={`text-2xl font-bold ${
                        i === 0 ? 'text-yellow-400' : 
                        i === 1 ? 'text-gray-400' :
                        i === 2 ? 'text-orange-400' : 'text-gray-500'
                      }`}>
                        #{i + 1}
                      </div>
                      <div>
                        <p className="text-[var(--color-text)] font-medium">{assoc.entraineur}</p>
                        <p className="text-xs text-gray-400">
                          {assoc.victoires} V / {assoc.courses} C • <span className="text-green-400">{assoc.taux}%</span>
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          )}
        </motion.div>
      )}

      {/* État initial */}
      {!jockeyStats && !loading && (
        <GlassCard className="text-center py-12">
          <div className="text-6xl mb-4">🏇</div>
          <p className="text-gray-400 text-lg">Sélectionnez un jockey pour voir ses statistiques complètes</p>
        </GlassCard>
      )}
    </div>
  );
}

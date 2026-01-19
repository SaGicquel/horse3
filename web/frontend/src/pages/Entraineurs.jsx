import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  UserGroupIcon,
  TrophyIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  MagnifyingGlassIcon,
  FireIcon
} from '@heroicons/react/24/outline';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import GlassCard from '../components/GlassCard';
import EntityHeader from '../components/EntityHeader';
import { useNavigate, useParams } from 'react-router-dom';
import { API_BASE } from '../config/api';

export default function Entraineurs() {
  const [entraineurs, setEntraineurs] = useState([]);
  const [topEntraineurs, setTopEntraineurs] = useState([]);
  const [selectedEntraineur, setSelectedEntraineur] = useState(null);
  const [entraineurStats, setEntraineurStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();
  const { id } = useParams();
  const formatCurrency = (value = 0) => `${Math.round((value || 0) / 100).toLocaleString('fr-FR')}€`;
  const formatPercent = (value) => value === undefined || value === null ? '--' : `${Number(value).toFixed(1)}%`;

  useEffect(() => {
    loadTopEntraineurs();
  }, []);

  const loadTopEntraineurs = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/entraineurs/top`);
      if (response.ok) {
        const data = await response.json();
        setTopEntraineurs(data.entraineurs || []);
        setEntraineurs(data.entraineurs || []);
      }
    } catch (error) {
      console.error('Erreur chargement entraineurs:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadEntraineurStats = async (nom) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/entraineur/${encodeURIComponent(nom)}/stats`);
      if (response.ok) {
        const data = await response.json();
        setEntraineurStats(data);
        setSelectedEntraineur(nom);
      }
    } catch (error) {
      console.error('Erreur chargement stats entraineur:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) {
      const decoded = decodeURIComponent(id);
      setSearchTerm(decoded);
      loadEntraineurStats(decoded);
    }
  }, [id]);

  const filteredEntraineurs = entraineurs.filter(e =>
    e.nom?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Données graphiques
  const tendanceData = entraineurStats?.tendance_mensuelle?.map(t => ({
    mois: t.mois,
    victoires: t.victoires,
    courses: t.courses,
    taux: t.taux
  })) || [];

  const typeData = entraineurStats?.performance_par_type?.map(p => ({
    type: p.type,
    victoires: p.victoires,
    courses: p.courses,
    taux: p.taux
  })) || [];

  const radarData = entraineurStats?.profil_specialisation || [];

  const recentMetrics = entraineurStats?.recent_metrics || {};

  const trendData = useMemo(() => {
    if (entraineurStats?.recent_performance?.length) {
      return entraineurStats.recent_performance.map((entry) => ({
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
  }, [entraineurStats, tendanceData]);

  const headerMetrics = entraineurStats ? [
    { label: 'Taux réussite', value: formatPercent(entraineurStats.statistiques.taux_victoire) },
    { label: 'Gains', value: formatCurrency(entraineurStats.statistiques.gains_total || 0) },
    { label: 'Forme 30j', value: formatPercent(recentMetrics.taux_victoire_30j ?? entraineurStats.statistiques.taux_victoire) },
    { label: 'Courses 30j', value: (recentMetrics.courses_30j ?? entraineurStats.statistiques.courses)?.toString() || '0' }
  ] : [];

  const handleSeeCourses = () => {
    const entraineurName = entraineurStats?.nom || selectedEntraineur || searchTerm;
    navigate(entraineurName ? `/courses?entraineur=${encodeURIComponent(entraineurName)}` : '/courses');
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent mb-2">
          👔 Entraineurs
        </h1>
        <p className="text-gray-400">
          Statistiques et performances des entraineurs
        </p>
      </motion.div>

      {/* Recherche */}
      <div className="flex justify-center">
        <div className="relative w-full max-w-md">
          <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Rechercher un entraineur..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-white/5 dark:bg-white/5 border border-white/10 dark:border-white/10 rounded-xl text-[var(--color-text)] placeholder-gray-500 focus:outline-none focus:border-orange-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Top Entraineurs */}
      <GlassCard>
        <div className="flex items-center gap-2 mb-4">
          <FireIcon className="h-6 w-6 text-orange-400" />
          <h3 className="text-lg font-semibold text-[var(--color-text)]">Top Entraineurs</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {(searchTerm ? filteredEntraineurs : topEntraineurs).slice(0, 12).map((entraineur, index) => (
            <motion.div
              key={entraineur.nom}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className={`bg-white/5 rounded-xl p-4 cursor-pointer hover:bg-white/10 transition-all ${
                selectedEntraineur === entraineur.nom ? 'ring-2 ring-orange-500' : ''
              }`}
              onClick={() => loadEntraineurStats(entraineur.nom)}
            >
              <div className="text-center">
                <div className="text-3xl mb-2">
                  {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '👔'}
                </div>
                <h4 className="text-[var(--color-text)] font-medium text-sm truncate">{entraineur.nom}</h4>
                <div className="flex justify-center gap-2 mt-2 text-xs">
                  <span className="text-green-400">{entraineur.victoires} V</span>
                  <span className="text-gray-400">{entraineur.taux}%</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </GlassCard>

      {/* Stats de l'entraineur sélectionné */}
      {entraineurStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <EntityHeader
            name={entraineurStats.nom}
            emoji="👔"
            subtitle="Synthèse des 30 derniers jours"
            pill="Entraineur"
            metrics={headerMetrics}
            trendData={trendData}
            onCta={handleSeeCourses}
            ctaLabel="Voir courses liées"
          />

          {/* Stats principales */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <GlassCard className="text-center">
              <TrophyIcon className="h-8 w-8 mx-auto text-yellow-400 mb-2" />
              <div className="text-2xl font-bold text-[var(--color-text)]">{entraineurStats.statistiques.victoires}</div>
              <div className="text-xs text-gray-400">Victoires</div>
            </GlassCard>

            <GlassCard className="text-center">
              <ChartBarIcon className="h-8 w-8 mx-auto text-blue-400 mb-2" />
              <div className="text-2xl font-bold text-[var(--color-text)]">{entraineurStats.statistiques.courses}</div>
              <div className="text-xs text-gray-400">Courses</div>
            </GlassCard>

            <GlassCard className="text-center">
              <ArrowTrendingUpIcon className="h-8 w-8 mx-auto text-green-400 mb-2" />
              <div className="text-2xl font-bold text-green-400">{entraineurStats.statistiques.taux_victoire}%</div>
              <div className="text-xs text-gray-400">Taux Victoire</div>
            </GlassCard>

            <GlassCard className="text-center">
              <div className="text-3xl mb-2">🐴</div>
              <div className="text-2xl font-bold text-purple-400">{entraineurStats.statistiques.nb_chevaux}</div>
              <div className="text-xs text-gray-400">Chevaux</div>
            </GlassCard>

            <GlassCard className="text-center">
              <div className="text-3xl mb-2">💰</div>
              <div className="text-2xl font-bold text-emerald-400">{entraineurStats.statistiques.gains_total}</div>
              <div className="text-xs text-gray-400">Gains Total</div>
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
                    dataKey="victoires"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    dot={{ fill: '#F59E0B', r: 4 }}
                    name="Victoires"
                  />
                </LineChart>
              </ResponsiveContainer>
            </GlassCard>

            {/* Performance par type */}
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🏁 Par Type de Course</h3>
              {typeData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={typeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="type" stroke="#9CA3AF" fontSize={10} />
                    <YAxis stroke="#9CA3AF" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="taux" fill="#F59E0B" name="Taux %" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex flex-col items-center justify-center h-[250px] text-gray-500">
                  <div className="text-4xl mb-2">📊</div>
                  <p>Données non disponibles</p>
                </div>
              )}
            </GlassCard>

            {/* Profil spécialisation */}
            {radarData.length > 0 && (
              <GlassCard>
                <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🎯 Profil de Spécialisation</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#374151" />
                    <PolarAngleAxis dataKey="categorie" stroke="#9CA3AF" fontSize={10} />
                    <PolarRadiusAxis stroke="#9CA3AF" fontSize={10} />
                    <Radar
                      name="Performance"
                      dataKey="score"
                      stroke="#F59E0B"
                      fill="#F59E0B"
                      fillOpacity={0.3}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </GlassCard>
            )}
          </div>

          {/* Top chevaux de l'écurie */}
          {entraineurStats.top_chevaux && entraineurStats.top_chevaux.length > 0 && (
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🐴 Top Chevaux de l'Écurie</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {entraineurStats.top_chevaux.slice(0, 6).map((cheval, i) => (
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
                        <p className="text-[var(--color-text)] font-medium">{cheval.nom}</p>
                        <p className="text-xs text-gray-400">
                          {cheval.victoires} V / {cheval.courses} C •
                          <span className="text-green-400 ml-1">{cheval.taux}%</span>
                        </p>
                        <p className="text-xs text-purple-400">Gains: {cheval.gains}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          )}

          {/* Jockeys favoris */}
          {entraineurStats.jockeys_favoris && entraineurStats.jockeys_favoris.length > 0 && (
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🤝 Jockeys Favoris</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {entraineurStats.jockeys_favoris.slice(0, 6).map((jockey, i) => (
                  <div key={i} className="bg-white/5 rounded-lg p-4 flex items-center gap-3">
                    <div className="text-3xl">👨‍✈️</div>
                    <div>
                      <p className="text-[var(--color-text)] font-medium">{jockey.nom}</p>
                      <p className="text-xs text-gray-400">
                        {jockey.victoires} V / {jockey.courses} C •
                        <span className="text-green-400 ml-1">{jockey.taux}%</span>
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          )}
        </motion.div>
      )}

      {/* État initial */}
      {!entraineurStats && !loading && (
        <GlassCard className="text-center py-12">
          <div className="text-6xl mb-4">🏠</div>
          <p className="text-gray-400 text-lg">Sélectionnez un entraineur pour voir ses statistiques complètes</p>
        </GlassCard>
      )}
    </div>
  );
}

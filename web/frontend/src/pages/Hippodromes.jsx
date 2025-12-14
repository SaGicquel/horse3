import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  MapPinIcon,
  TrophyIcon,
  ChartBarIcon,
  CalendarIcon,
  UserGroupIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import GlassCard from '../components/GlassCard';
import EntityHeader from '../components/EntityHeader';
import { useNavigate, useParams } from 'react-router-dom';
import { API_BASE } from '../config/api';

export default function Hippodromes() {
  const [hippodromes, setHippodromes] = useState([]);
  const [selectedHippodrome, setSelectedHippodrome] = useState(null);
  const [hippodromeStats, setHippodromeStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStats, setLoadingStats] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();
  const { id } = useParams();
  const formatCurrency = (value = 0) => `${Math.round((value || 0) / 100).toLocaleString('fr-FR')}€`;
  const formatPercent = (value) => value === undefined || value === null ? '--' : `${Number(value).toFixed(1)}%`;

  useEffect(() => {
    loadHippodromes();
  }, []);

  const loadHippodromes = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/hippodromes`);
      if (response.ok) {
        const data = await response.json();
        setHippodromes(data.hippodromes || []);
      }
    } catch (error) {
      console.error('Erreur chargement hippodromes:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadHippodromeStats = async (nom) => {
    // Mettre à jour l'hippodrome sélectionné immédiatement
    setSelectedHippodrome(nom);
    setLoadingStats(true);
    setHippodromeStats(null); // Reset pour forcer le re-render
    
    try {
      const response = await fetch(`${API_BASE}/api/hippodrome/${encodeURIComponent(nom)}/stats`);
      if (response.ok) {
        const data = await response.json();
        setHippodromeStats(data);
      } else {
        console.error('Erreur réponse:', response.status);
      }
    } catch (error) {
      console.error('Erreur chargement stats hippodrome:', error);
    } finally {
      setLoadingStats(false);
    }
  };

  useEffect(() => {
    if (id) {
      const decoded = decodeURIComponent(id);
      setSelectedHippodrome(decoded);
      setSearchTerm(decoded);
      loadHippodromeStats(decoded);
    }
  }, [id]);

  // Extraire le nom court de l'hippodrome (sans "HIPPODROME DE")
  const getShortName = (nom) => {
    return nom
      .replace(/^HIPPODROME D[E']?\s*/i, '')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const filteredHippodromes = hippodromes.filter(h => 
    h.nom.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Préparer données graphiques
  const performanceData = hippodromeStats?.performance_par_type?.map(p => ({
    type: p.type,
    courses: p.courses,
    favoris: p.taux_favoris,
    outsiders: p.taux_outsiders
  })) || [];

  const distanceData = hippodromeStats?.statistiques_distances?.map(d => ({
    distance: `${d.distance}m`,
    courses: d.courses,
    taux_favori: d.taux_favori
  })) || [];

  const recentMetrics = hippodromeStats?.recent_metrics || {};

  const trendData = useMemo(() => {
    if (hippodromeStats?.recent_performance?.length) {
      return hippodromeStats.recent_performance.map((entry) => ({
        label: entry.date?.slice(5),
        value: Number(entry.taux ?? 0)
      }));
    }
    return [];
  }, [hippodromeStats]);

  const headerMetrics = hippodromeStats ? [
    { label: 'Taux réussite', value: formatPercent(hippodromeStats.taux_favoris) },
    { label: 'Gains', value: formatCurrency(hippodromeStats.gains_total || 0) },
    { label: 'Forme 30j', value: formatPercent(recentMetrics.taux_victoire_30j ?? hippodromeStats.taux_favoris) },
    { label: 'Courses 30j', value: (recentMetrics.courses_30j ?? hippodromeStats.total_courses)?.toString() || '0' }
  ] : [];

  const handleSeeCourses = () => {
    const hippoName = hippodromeStats?.nom || selectedHippodrome;
    navigate(hippoName ? `/courses?hippodrome=${encodeURIComponent(hippoName)}` : '/courses');
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-2">
          🏇 Hippodromes
        </h1>
        <p className="text-gray-400">
          Analysez les statistiques par hippodrome
        </p>
      </motion.div>

      {/* Recherche */}
      <div className="flex justify-center">
        <div className="relative w-full max-w-md">
          <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Rechercher un hippodrome..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-white/5 dark:bg-white/5 border border-white/10 dark:border-white/10 rounded-xl text-[var(--color-text)] placeholder-gray-500 focus:outline-none focus:border-green-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="h-28 bg-white/5 rounded-xl animate-pulse" />
          ))}
        </div>
      )}

      {/* Grid des hippodromes - utiliser les vrais hippodromes de la BDD */}
      {!loading && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {filteredHippodromes.map((hippodrome, index) => (
            <motion.div
              key={hippodrome.nom}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.03 }}
            >
              <GlassCard 
                className={`cursor-pointer hover:border-green-500/50 hover:scale-[1.02] transition-all duration-200 ${
                  selectedHippodrome === hippodrome.nom ? 'border-green-500/50 ring-2 ring-green-500/30 bg-green-500/10' : ''
                }`}
                onClick={() => loadHippodromeStats(hippodrome.nom)}
              >
                <div className="text-center p-2">
                  <div className="text-2xl mb-2">🏟️</div>
                  <h3 className="font-medium text-[var(--color-text)] text-xs leading-tight line-clamp-2">
                    {getShortName(hippodrome.nom)}
                  </h3>
                  <p className="text-xs text-green-400 mt-2 font-semibold">{hippodrome.courses} courses</p>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      )}

      {/* Stats de l'hippodrome sélectionné */}
      {selectedHippodrome && loadingStats && (
        <GlassCard className="text-center py-12">
          <div className="animate-spin text-6xl mb-4">🏇</div>
          <p className="text-gray-400 dark:text-gray-400">Chargement des statistiques de {getShortName(selectedHippodrome)}...</p>
        </GlassCard>
      )}

      {hippodromeStats && !loadingStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <EntityHeader
            name={hippodromeStats.nom}
            emoji="🏟️"
            subtitle="Synthèse des 30 derniers jours"
            pill="Hippodrome"
            metrics={headerMetrics}
            trendData={trendData}
            onCta={handleSeeCourses}
            ctaLabel="Voir courses liées"
          />

          {/* Stats principales */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <GlassCard className="text-center">
              <TrophyIcon className="h-8 w-8 mx-auto text-yellow-400 mb-2" />
              <div className="text-2xl font-bold text-[var(--color-text)]">{hippodromeStats.total_courses}</div>
              <div className="text-xs text-gray-400">Courses</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <ChartBarIcon className="h-8 w-8 mx-auto text-blue-400 mb-2" />
              <div className="text-2xl font-bold text-blue-400">{hippodromeStats.taux_favoris}%</div>
              <div className="text-xs text-gray-400">Favoris gagnants</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <div className="text-3xl mb-2">💰</div>
              <div className="text-2xl font-bold text-green-400">{hippodromeStats.cote_moyenne}</div>
              <div className="text-xs text-gray-400">Cote moyenne gagnant</div>
            </GlassCard>
            
            <GlassCard className="text-center">
              <CalendarIcon className="h-8 w-8 mx-auto text-purple-400 mb-2" />
              <div className="text-2xl font-bold text-purple-400">{hippodromeStats.derniere_course}</div>
              <div className="text-xs text-gray-400">Dernière course</div>
            </GlassCard>
          </div>

          {/* Graphiques */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Performance par type */}
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">📊 Performance par Type</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={performanceData}>
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
                  <Bar dataKey="favoris" fill="#10B981" name="% Favoris" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="outsiders" fill="#8B5CF6" name="% Outsiders" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </GlassCard>

            {/* Performance par distance */}
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">📏 Performance par Distance</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={distanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="distance" stroke="#9CA3AF" fontSize={10} />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="taux_favori" fill="#F59E0B" name="% Favori gagne" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </GlassCard>
          </div>

          {/* Top performers */}
          {hippodromeStats.top_chevaux && hippodromeStats.top_chevaux.length > 0 && (
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🏆 Top Chevaux sur cet Hippodrome</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {hippodromeStats.top_chevaux.slice(0, 6).map((cheval, i) => (
                  <div key={i} className="bg-white/5 rounded-lg p-3 flex items-center gap-3">
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
                        {cheval.victoires} V / {cheval.courses} C • {cheval.taux}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          )}

          {/* Top jockeys */}
          {hippodromeStats.top_jockeys && hippodromeStats.top_jockeys.length > 0 && (
            <GlassCard>
              <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">👨‍✈️ Top Jockeys sur cet Hippodrome</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {hippodromeStats.top_jockeys.slice(0, 6).map((jockey, i) => (
                  <div key={i} className="bg-white/5 rounded-lg p-3 flex items-center gap-3">
                    <div className={`text-2xl font-bold ${
                      i === 0 ? 'text-yellow-400' : 
                      i === 1 ? 'text-gray-400' :
                      i === 2 ? 'text-orange-400' : 'text-gray-500'
                    }`}>
                      #{i + 1}
                    </div>
                    <div>
                      <p className="text-[var(--color-text)] font-medium">{jockey.nom}</p>
                      <p className="text-xs text-gray-400">
                        {jockey.victoires} V / {jockey.courses} C • {jockey.taux}%
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
      {!selectedHippodrome && !loading && (
        <GlassCard className="text-center py-12">
          <div className="text-6xl mb-4">🏟️</div>
          <p className="text-gray-400 text-lg">Sélectionnez un hippodrome pour voir ses statistiques</p>
        </GlassCard>
      )}
    </div>
  );
}

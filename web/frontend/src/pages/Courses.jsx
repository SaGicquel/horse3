import { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useSearchParams } from 'react-router-dom';
import {
  CalendarIcon,
  MapPinIcon,
  ClockIcon,
  UserGroupIcon,
  TrophyIcon,
  ChevronRightIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  ArrowPathIcon,
  InformationCircleIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  XMarkIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import PageHeader from '../components/PageHeader';
import { API_BASE } from '../config/api';
import { percent, odds as formatOdds } from '../lib/format';
import { exportPicksCSV, exportPicksJSON } from '../lib/export';
import { getBucketClassNames, normalizeBucketLabel } from '../lib/buckets';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.05 }
  }
};

const itemVariants = {
  hidden: { y: 10, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 100, damping: 15 }
  }
};

// Tooltip component pour le rationale
const RationaleTooltip = ({ rationale, children }) => {
  const [show, setShow] = useState(false);

  if (!rationale || rationale.length === 0) return children;

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        {children}
      </div>
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ opacity: 0, y: 5, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.95 }}
            className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-4 py-3 rounded-xl min-w-[250px] max-w-[350px] glass-panel border border-neutral-200 dark:border-white/20 shadow-xl"
          >
            <ul className="space-y-1.5 text-neutral-800 dark:text-neutral-200 text-xs">
              {rationale.slice(0, 5).map((item, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-primary-400 mt-0.5">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-neutral-100 dark:bg-neutral-900 border-r border-b border-neutral-200 dark:border-white/20" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function Courses() {
  const [courses, setCourses] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [selectedCourseData, setSelectedCourseData] = useState(null);
  const [raceAnalysis, setRaceAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [filterDiscipline, setFilterDiscipline] = useState('all');
  const [valueThreshold, setValueThreshold] = useState(0);
  const [searchParams, setSearchParams] = useSearchParams();
  const [settings, setSettings] = useState(null);

  // Charger les settings pour value_cutoff
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/settings`);
        if (response.ok) {
          const data = await response.json();
          setSettings(data);
          // Initialiser valueThreshold avec value_cutoff si disponible
          const cutoff = data?.betting_defaults?.value_cutoff;
          if (cutoff !== undefined) {
            setValueThreshold(cutoff * 100); // Convertir en pourcentage
          }
        }
      } catch (error) {
        console.error('Erreur chargement settings:', error);
      }
    };
    fetchSettings();
  }, []);

  // Paramètres de la politique de mise
  const valueCutoff = settings?.betting_defaults?.value_cutoff || 0.05;
  const capPerBet = settings?.betting_defaults?.cap_per_bet || 0.02;

  useEffect(() => {
    fetchCourses();
  }, []);

  const normalizeText = (value) => {
    if (!value) return '';
    return value
      .toString()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase();
  };

  const contextFilters = useMemo(() => ({
    hippodrome: searchParams.get('hippodrome'),
    cheval: searchParams.get('cheval'),
    jockey: searchParams.get('jockey'),
    entraineur: searchParams.get('entraineur')
  }), [searchParams]);

  const hasContextFilters = useMemo(
    () => Object.values(contextFilters).some(Boolean),
    [contextFilters]
  );

  const matchesContextFilters = useCallback((course) => {
    if (!hasContextFilters) return true;

    const targetHippo = contextFilters.hippodrome ? normalizeText(contextFilters.hippodrome) : null;
    if (targetHippo) {
      const courseHippo = normalizeText(course.hippodrome || course.venue || course.hippodrome_nom);
      if (!courseHippo.includes(targetHippo)) {
        return false;
      }
    }

    const requiresParticipantMatch = contextFilters.cheval || contextFilters.jockey || contextFilters.entraineur;
    if (requiresParticipantMatch) {
      const rawParticipants = course.participants || course.horses || course.chevaux || course.runners || course.selection || [];
      const participants = Array.isArray(rawParticipants) ? rawParticipants : Object.values(rawParticipants || {});

      if (participants.length > 0) {
        const targetCheval = contextFilters.cheval ? normalizeText(contextFilters.cheval) : null;
        const targetJockey = contextFilters.jockey ? normalizeText(contextFilters.jockey) : null;
        const targetEnt = contextFilters.entraineur ? normalizeText(contextFilters.entraineur) : null;

        const hasMatch = participants.some((runner) => {
          const chevalName = normalizeText(runner.nom || runner.name || runner.cheval || runner.idCheval);
          const jockeyName = normalizeText(runner.driver || runner.jockey || runner.driver_jockey);
          const entraineurName = normalizeText(runner.entraineur || runner.trainer);

          const chevalOk = targetCheval ? chevalName.includes(targetCheval) : true;
          const jockeyOk = targetJockey ? jockeyName.includes(targetJockey) : true;
          const entOk = targetEnt ? entraineurName.includes(targetEnt) : true;
          return chevalOk && jockeyOk && entOk;
        });

        if (!hasMatch) {
          return false;
        }
      }
    }

    return true;
  }, [contextFilters, hasContextFilters]);

  const clearContextFilters = useCallback(() => {
    const next = new URLSearchParams(searchParams);
    ['hippodrome', 'cheval', 'jockey', 'entraineur'].forEach((key) => next.delete(key));
    setSearchParams(next);
  }, [searchParams, setSearchParams]);

  const fetchCourses = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/picks/today`);
      if (response.ok) {
        const data = await response.json();
        // Normaliser les données - peut être un array direct ou un objet avec .courses/.races
        const coursesArray = Array.isArray(data)
          ? data
          : (data.courses || data.races || data.picks || []);
        setCourses(coursesArray);
      } else {
        setCourses([]);
      }
    } catch (error) {
      console.error('Erreur chargement courses:', error);
      setCourses([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchRaceAnalysis = useCallback(async (raceKey, courseData) => {
    setLoadingAnalysis(true);
    setSelectedCourse(raceKey);
    setSelectedCourseData(courseData);
    try {
      const response = await fetch(`${API_BASE}/analyze/${encodeURIComponent(raceKey)}`);
      if (response.ok) {
        const data = await response.json();
        setRaceAnalysis(data);
      } else {
        setRaceAnalysis(null);
      }
    } catch (error) {
      console.error('Erreur chargement analyse:', error);
      setRaceAnalysis(null);
    } finally {
      setLoadingAnalysis(false);
    }
  }, []);

  const closeRacePanel = () => {
    setSelectedCourse(null);
    setSelectedCourseData(null);
    setRaceAnalysis(null);
  };

  // Extraire les disciplines uniques
  const disciplines = useMemo(() => {
    const types = new Set();
    courses.forEach(c => {
      const discipline = c.discipline || c.type_course || c.specialite;
      if (discipline) types.add(discipline);
    });
    return [...types].sort();
  }, [courses]);

  // Filtrer les courses
  const filteredCourses = useMemo(() => {
    return courses.filter(course => {
      const discipline = course.discipline || course.type_course || course.specialite || '';
      const matchesDiscipline = filterDiscipline === 'all' || discipline === filterDiscipline;
      const matchesContext = matchesContextFilters(course);
      return matchesDiscipline && matchesContext;
    });
  }, [courses, filterDiscipline, matchesContextFilters]);

  // Trier les chevaux de l'analyse par value desc, kelly desc
  // Note: On affiche tous les chevaux, le filtrage visuel se fait via l'indicateur de value
  const sortedHorses = useMemo(() => {
    if (!raceAnalysis) return [];
    const horses = raceAnalysis.analyses || raceAnalysis.runners || raceAnalysis.horses || raceAnalysis.participants || raceAnalysis.chevaux || [];

    return horses
      .sort((a, b) => {
        const valueA = a.value ?? a.value_pct ?? a.value_bet ?? 0;
        const valueB = b.value ?? b.value_pct ?? b.value_bet ?? 0;
        if (valueB !== valueA) return valueB - valueA;
        // Tie-break par kelly desc
        const kellyA = a.kelly ?? a.kelly_pct ?? 0;
        const kellyB = b.kelly ?? b.kelly_pct ?? 0;
        return kellyB - kellyA;
      });
  }, [raceAnalysis]);

  const exportCurrentRace = useCallback((format = 'csv') => {
    if (!raceAnalysis) return;
    const picks = sortedHorses.map((horse) => ({
      ...horse,
      hippodrome: selectedCourseData?.hippodrome || selectedCourseData?.venue,
      race_key: selectedCourse || raceAnalysis.race_key || raceAnalysis.raceKey,
    }));

    const meta = {
      race: selectedCourse || raceAnalysis?.race_key || raceAnalysis?.raceKey,
      hippodrome: selectedCourseData?.hippodrome || selectedCourseData?.venue,
      exported: new Date().toISOString(),
    };

    if (format === 'json') {
      exportPicksJSON(picks, { filename: `race_${selectedCourse || 'picks'}`, meta });
    } else {
      exportPicksCSV(picks, { filename: `race_${selectedCourse || 'picks'}` });
    }
  }, [raceAnalysis, selectedCourse, selectedCourseData, sortedHorses]);

  // Vérifier si la colonne drift est disponible
  const hasDrift = useMemo(() => {
    if (!raceAnalysis) return false;
    const horses = raceAnalysis.analyses || raceAnalysis.runners || raceAnalysis.horses || raceAnalysis.participants || raceAnalysis.chevaux || [];
    return horses.some(h => h.drift !== undefined && h.drift !== null);
  }, [raceAnalysis]);

  const getValueColor = (value) => {
    if (value >= 20) return 'text-green-600 dark:text-success';
    if (value >= 10) return 'text-emerald-600 dark:text-emerald-400';
    if (value >= 5) return 'text-amber-600 dark:text-warning';
    if (value >= 0) return 'text-neutral-600 dark:text-neutral-400';
    return 'text-red-600 dark:text-error';
  };

  const getDriftColor = (drift) => {
    if (drift > 0.5) return 'text-red-600 dark:text-error';
    if (drift > 0.1) return 'text-amber-600 dark:text-warning';
    if (drift < -0.5) return 'text-green-600 dark:text-success';
    if (drift < -0.1) return 'text-emerald-600 dark:text-emerald-400';
    return 'text-neutral-600 dark:text-neutral-400';
  };

  const formatTime = (time) => {
    if (!time) return '--:--';
    // Si c'est déjà au format HH:MM
    if (typeof time === 'string' && time.includes(':')) return time;
    // Sinon essayer de parser
    return time;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-3 sm:px-6 py-6 sm:py-12">
      {/* Header unifié */}
      <PageHeader
        emoji="🏇"
        title="Courses du Jour"
        subtitle="Analysez les courses et leurs probabilités"
      >
        {/* Filtres */}
        <select
          value={filterDiscipline}
          onChange={(e) => setFilterDiscipline(e.target.value)}
          className="glass-input px-4 py-2 rounded-xl"
        >
          <option value="all">Toutes disciplines</option>
          {disciplines.map(d => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>

        <div className="flex items-center gap-2 px-3 py-2 glass-panel border border-neutral-200 dark:border-white/10 rounded-xl">
          <label className="text-sm text-neutral-800 dark:text-neutral-400">Value ≥</label>
          <input
            type="number"
            value={valueThreshold}
            onChange={(e) => setValueThreshold(Number(e.target.value) || 0)}
            className="w-16 px-2 py-1 bg-transparent text-neutral-900 dark:text-neutral-100 focus:outline-none text-center"
            min={-100}
            max={100}
            aria-label="Value threshold"
          />
          <span className="text-neutral-500 dark:text-neutral-400">%</span>
        </div>

        <motion.button
          onClick={fetchCourses}
          className="p-2 glass-button hover:bg-neutral-100 dark:hover:bg-white/10"
          title="Actualiser"
        >
          <ArrowPathIcon className="h-5 w-5 text-neutral-500 dark:text-neutral-400" />
        </motion.button>
      </PageHeader>

      {hasContextFilters && (
        <div className="glass-panel border border-neutral-200 dark:border-white/10 rounded-xl px-3 py-2 flex flex-wrap items-center gap-2">
          <span className="text-xs text-neutral-700 dark:text-neutral-400">Filtre contextuel</span>
          {contextFilters.hippodrome && (
            <span className="px-2 py-1 rounded-lg bg-primary-500/10 text-primary-400 text-xs border border-primary-500/20">
              Hippodrome: {contextFilters.hippodrome}
            </span>
          )}
          {contextFilters.cheval && (
            <span className="px-2 py-1 rounded-lg bg-secondary-500/10 text-secondary-400 text-xs border border-secondary-500/20">
              Cheval: {contextFilters.cheval}
            </span>
          )}
          {contextFilters.jockey && (
            <span className="px-2 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 text-xs border border-emerald-500/20">
              Jockey: {contextFilters.jockey}
            </span>
          )}
          {contextFilters.entraineur && (
            <span className="px-2 py-1 rounded-lg bg-orange-500/10 text-orange-400 text-xs border border-orange-500/20">
              Entraineur: {contextFilters.entraineur}
            </span>
          )}
          <button
            onClick={clearContextFilters}
            className="text-xs px-2 py-1 rounded-lg hover:bg-neutral-100 dark:hover:bg-white/10 transition-colors text-neutral-700 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-200"
          >
            Réinitialiser
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Liste des courses */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-4"
        >
          <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
            <CalendarIcon className="h-5 w-5 text-primary-500" />
            Courses disponibles ({filteredCourses.length})
          </h2>

          {loading ? (
            <div className="space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-28 bg-neutral-200/50 dark:bg-neutral-800/50 rounded-xl animate-pulse" />
              ))}
            </div>
          ) : filteredCourses.length === 0 ? (
            <GlassCard className="text-center py-12 border border-warning/30" hover={false}>
              <div className="text-6xl mb-4">📅</div>
              <h3 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">Aucune course disponible</h3>
              <p className="text-neutral-700 dark:text-neutral-400 mb-4">
                Il n'y a pas de course programmée pour aujourd'hui.
              </p>
              <p className="text-sm text-neutral-700">
                Les courses sont généralement disponibles entre 10h et 22h.
              </p>
            </GlassCard>
          ) : (
            <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-2 custom-scrollbar">
              {filteredCourses.map((course) => {
                const raceKey = course.race_key || course.raceKey || course.id;
                const discipline = course.discipline || course.type_course || course.specialite || 'Course';
                const hippodrome = course.hippodrome || course.venue || 'Hippodrome';
                const hippodromeId = course.hippodrome_id || course.hippodromeId;
                const time = course.heure || course.time || course.depart;
                const nbPartants = course.nb_partants || course.partants || course.runners;

                return (
                  <motion.div
                    key={raceKey}
                    variants={itemVariants}
                  >
                    <GlassCard
                      className={`transition-all cursor-pointer ${selectedCourse === raceKey
                        ? 'ring-2 ring-primary-500 bg-primary-500/10'
                        : 'hover:bg-neutral-50 dark:hover:bg-white/5'
                        }`}
                      onClick={() => fetchRaceAnalysis(raceKey, course)}
                    >
                      <div className="flex justify-between items-center">
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="px-2 py-1 bg-secondary-500/20 text-secondary-400 rounded text-xs font-medium border border-secondary-500/20">
                              {discipline}
                            </span>
                            <span className="flex items-center gap-1 text-sm text-neutral-700 dark:text-neutral-400">
                              <ClockIcon className="h-4 w-4" />
                              {formatTime(time)}
                            </span>
                            {nbPartants && (
                              <span className="flex items-center gap-1 text-sm text-neutral-700 dark:text-neutral-400">
                                <UserGroupIcon className="h-4 w-4" />
                                {nbPartants}
                              </span>
                            )}
                          </div>

                          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                            <MapPinIcon className="h-5 w-5 text-primary-500 flex-shrink-0" />
                            {hippodromeId ? (
                              <Link
                                to={`/hippodrome/${hippodromeId}`}
                                className="hover:text-primary-400 transition-colors"
                                onClick={(e) => e.stopPropagation()}
                              >
                                {hippodrome}
                              </Link>
                            ) : (
                              hippodrome
                            )}
                          </h3>

                          {course.distance && (
                            <span className="text-sm text-neutral-800 dark:text-neutral-500">{course.distance}m</span>
                          )}
                        </div>

                        <button
                          className="flex items-center gap-2 px-4 py-2 glass-button-primary rounded-xl text-sm"
                        >
                          <EyeIcon className="h-4 w-4" />
                          Voir
                        </button>
                      </div>
                    </GlassCard>
                  </motion.div>
                );
              })}
            </div>
          )}
        </motion.div>

        {/* Panel Race Analysis */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-4"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
              <TrophyIcon className="h-5 w-5 text-secondary-400" />
              Analyse Course
            </h2>
            {selectedCourse && (
              <motion.button
                onClick={closeRacePanel}
                className="p-1 hover:bg-neutral-100 dark:hover:bg-white/10 rounded-lg transition-colors"
              >
                <XMarkIcon className="h-5 w-5 text-neutral-500 dark:text-neutral-400" />
              </motion.button>
            )}
          </div>

          {!selectedCourse ? (
            <GlassCard className="text-center py-12" hover={false}>
              <div className="text-6xl mb-4">📊</div>
              <p className="text-neutral-700 dark:text-neutral-400">
                Cliquez sur "Voir" pour analyser une course
              </p>
            </GlassCard>
          ) : loadingAnalysis ? (
            <div className="space-y-3">
              <div className="h-12 bg-neutral-200/50 dark:bg-neutral-800/50 rounded-xl animate-pulse" />
              {[...Array(6)].map((_, i) => (
                <div key={i} className="h-16 bg-neutral-200/50 dark:bg-neutral-800/50 rounded-xl animate-pulse" />
              ))}
            </div>
          ) : !raceAnalysis ? (
            <GlassCard className="text-center py-12 border border-warning/30" hover={false}>
              <div className="text-6xl mb-4">⚠️</div>
              <p className="text-neutral-700 dark:text-neutral-400">Analyse non disponible pour cette course</p>
            </GlassCard>
          ) : (
            <div className="space-y-4">
              {/* Header de la course sélectionnée */}
              {selectedCourseData && (
                <GlassCard className="p-4" hover={false}>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-neutral-700 dark:text-neutral-400">
                        {selectedCourseData.discipline || selectedCourseData.type_course || 'Course'}
                      </span>
                      <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 text-lg">
                        {selectedCourseData.hippodrome || selectedCourseData.venue}
                      </h3>
                    </div>
                    <span className="text-2xl font-bold text-primary-500">
                      {formatTime(selectedCourseData.heure || selectedCourseData.time)}
                    </span>
                  </div>
                </GlassCard>
              )}

              <div className="flex flex-wrap gap-2 justify-end">
                <motion.button
                  onClick={() => exportCurrentRace('csv')}
                  className="glass-button px-3 py-1.5 text-xs"
                >
                  Export CSV
                </motion.button>
                <motion.button
                  onClick={() => exportCurrentRace('json')}
                  className="glass-button px-3 py-1.5 text-xs"
                >
                  Export JSON
                </motion.button>
              </div>

              {/* Tableau des chevaux */}
              <div className="overflow-x-auto rounded-xl border border-neutral-200 dark:border-white/10">
                <table className="w-full text-sm">
                  <thead className="bg-neutral-100 dark:bg-white/5">
                    <tr className="border-b border-neutral-200 dark:border-white/10">
                      <th className="text-left py-3 px-3 text-neutral-800 dark:text-neutral-400 font-medium">Cheval</th>
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">p(win)</th>
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">Value</th>
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">Kelly</th>
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">Cote</th>
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">Bucket</th>
                      {hasDrift && (
                        <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">Drift</th>
                      )}
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">
                        <InformationCircleIcon className="h-4 w-4 inline" />
                      </th>
                      <th className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400 font-medium">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-200 dark:divide-white/5">
                    <AnimatePresence>
                      {sortedHorses.map((horse, index) => {
                        const horseName = horse.nom || horse.name || horse.cheval || `Cheval ${index + 1}`;
                        const horseId = horse.cheval_id || horse.chevalId || horse.id;
                        const pWin = horse.p_win ?? horse.proba_win ?? horse.probabilite ?? 0;
                        const pPlace = horse.p_place ?? horse.proba_place ?? 0;
                        const value = horse.value ?? horse.value_pct ?? horse.value_bet ?? 0;
                        const kelly = horse.kelly ?? horse.kelly_pct ?? 0;
                        const fairOdds = horse.fair ?? horse.fair_odds ?? horse.cote_fair ?? 0;
                        const marketOdds = horse.market ?? horse.cote ?? horse.odds ?? 0;
                        const bucket = horse.bucket ?? horse.classe ?? '';
                        const bucketLabel = normalizeBucketLabel(bucket);
                        const bucketClasses = getBucketClassNames(bucket);
                        const drift = horse.drift;
                        const rationale = horse.rationale || horse.raisons || horse.reasons || [];

                        // Vérifier si value >= seuil pour autoriser le panier
                        // value est en % (ex: 5.2 pour 5.2%), valueThreshold aussi
                        const meetsValueCutoff = value >= valueThreshold;

                        return (
                          <motion.tr
                            key={horseId || horseName}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ delay: index * 0.03 }}
                            className="hover:bg-neutral-50 dark:hover:bg-white/5 transition-colors"
                          >
                            <td className="py-3 px-3">
                              <div className="flex items-center gap-3">
                                <span className="w-6 h-6 rounded-full bg-gradient-to-br from-[#ec4899] to-[#be185d] flex items-center justify-center text-white text-xs font-bold shadow-lg shadow-[#ec489933]">
                                  {horse.numero || index + 1}
                                </span>
                                {horseId ? (
                                  <Link
                                    to={`/cheval/${horseId}`}
                                    className="font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                                  >
                                    {horseName}
                                  </Link>
                                ) : (
                                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{horseName}</span>
                                )}
                              </div>
                            </td>
                            <td className="text-center py-3 px-2 text-primary-600 dark:text-primary-400 font-medium">
                              {percent(pWin)}
                            </td>
                            <td className={`text-center py-3 px-2 font-bold ${getValueColor(value)}`}>
                              {value > 0 ? `+${percent(value)}` : percent(value)}
                            </td>
                            <td className="text-center py-3 px-2 text-amber-600 dark:text-warning">
                              {kelly > 0 ? percent(kelly) : '-'}
                            </td>
                            <td className="text-center py-3 px-2 text-neutral-800 dark:text-neutral-400">
                              {marketOdds > 0 ? formatOdds(marketOdds) : '-'}
                            </td>
                            <td className="text-center py-3 px-2">
                              {bucketLabel && (
                                <span
                                  className={`px-2 py-0.5 rounded text-xs ${bucketClasses}`}
                                  data-testid="bucket-chip"
                                >
                                  {bucketLabel}
                                </span>
                              )}
                            </td>
                            {hasDrift && (
                              <td className={`text-center py-3 px-2 font-medium ${getDriftColor(drift)}`}>
                                {drift !== undefined && drift !== null ? (
                                  <span className="flex items-center justify-center gap-1">
                                    {drift > 0 ? <ChevronUpIcon className="h-3 w-3" /> : drift < 0 ? <ChevronDownIcon className="h-3 w-3" /> : null}
                                    {Math.abs(drift).toFixed(2)}
                                  </span>
                                ) : '-'}
                              </td>
                            )}
                            <td className="text-center py-3 px-2">
                              {rationale.length > 0 ? (
                                <RationaleTooltip rationale={rationale}>
                                  <InformationCircleIcon className="h-4 w-4 text-primary-600 dark:text-primary-400 cursor-help mx-auto" />
                                </RationaleTooltip>
                              ) : (
                                <span className="text-neutral-700 dark:text-neutral-600">-</span>
                              )}
                            </td>
                            <td className="text-center py-3 px-2">
                              {meetsValueCutoff ? (
                                <Link
                                  to={`/conseils?add=${horseId || horseName}&race=${selectedCourse}`}
                                  className="px-2 py-1 bg-success/10 text-success border border-success/20 rounded text-xs hover:bg-success/20 transition-colors whitespace-nowrap"
                                  title={`Cap: ${(capPerBet * 100).toFixed(0)}% bankroll par pari`}
                                >
                                  + Panier
                                </Link>
                              ) : (
                                <span
                                  className="px-2 py-1 bg-neutral-200 dark:bg-neutral-500/10 text-neutral-700 rounded text-xs cursor-not-allowed border border-neutral-300 dark:border-neutral-500/10 whitespace-nowrap"
                                  title={`Value ${value.toFixed(1)}% < seuil ${valueThreshold.toFixed(0)}%`}
                                >
                                  &lt; {valueThreshold.toFixed(0)}%
                                </span>
                              )}
                            </td>
                          </motion.tr>
                        );
                      })}
                    </AnimatePresence>
                  </tbody>
                </table>
              </div>

              {sortedHorses.length === 0 && (
                <div className="text-center py-8 text-neutral-700 dark:text-neutral-400">
                  Aucune analyse disponible pour cette course
                </div>
              )}

              {/* Légende */}
              <div className="flex flex-wrap gap-3 text-xs text-neutral-700 dark:text-neutral-500 pt-2 border-t border-neutral-200 dark:border-white/10">
                <span>p(win) = Probabilité de victoire</span>
                <span>•</span>
                <span>Value = Écart vs marché</span>
                <span>•</span>
                <span>Kelly = Mise optimale</span>
                <span>•</span>
                <span>Fair = Cote juste</span>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
